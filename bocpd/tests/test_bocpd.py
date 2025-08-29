# test_bocpd.py
# MIT License
# (c) 2025

import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from bocpd import (
    BOCPD,
    BOCPDConfig,
    BOCPDResult,
    ConstantHazard,
    ScheduledHazard,
    BoostedBoundaryHazard,
    Hazard,
)
from common.io.data_loader import parse_binary_string


class TestHazards(unittest.TestCase):
    """Test hazard function implementations."""

    def test_constant_hazard(self):
        h = ConstantHazard(mean_run_length=100.0)
        self.assertAlmostEqual(h.prob(10, 20), 0.01)
        self.assertAlmostEqual(h.prob(50, 100), 0.01)

        # Test extreme values are clamped
        h2 = ConstantHazard(mean_run_length=1e-6, eps=1e-5)
        self.assertAlmostEqual(h2.prob(0, 0), 1.0 - 1e-5)

    def test_scheduled_hazard(self):
        schedule = [0.1, 0.2, 0.3, 0.4]
        h = ScheduledHazard(schedule=schedule, period=4)

        # Check schedule is used correctly
        self.assertAlmostEqual(h.prob(0, 0), 0.1)
        self.assertAlmostEqual(h.prob(0, 1), 0.2)
        self.assertAlmostEqual(h.prob(0, 2), 0.3)
        self.assertAlmostEqual(h.prob(0, 3), 0.4)
        self.assertAlmostEqual(h.prob(0, 4), 0.1)  # Wraps around

        # Test validation
        with self.assertRaises(ValueError):
            ScheduledHazard(schedule=[0.1, 0.2], period=4)

    def test_boosted_boundary_hazard(self):
        base = ConstantHazard(mean_run_length=100.0)  # h = 0.01
        h = BoostedBoundaryHazard(
            base=base, period=4, boundary_indices=frozenset([0]), boost_factor=10.0
        )

        # Normal times
        self.assertAlmostEqual(h.prob(0, 1), 0.01)
        self.assertAlmostEqual(h.prob(0, 2), 0.01)
        self.assertAlmostEqual(h.prob(0, 3), 0.01)

        # Boundary times (boosted)
        self.assertAlmostEqual(h.prob(0, 0), 0.1)
        self.assertAlmostEqual(h.prob(0, 4), 0.1)

        # Test extreme values are clamped
        h2 = BoostedBoundaryHazard(
            base=ConstantHazard(mean_run_length=1e-6),
            period=4,
            boundary_indices=frozenset([0]),
            boost_factor=100.0,
            eps=1e-5,
        )
        self.assertAlmostEqual(h2.prob(0, 0), 1.0 - 1e-5)


class TestBOCPD(unittest.TestCase):
    """Test BOCPD algorithm functionality."""

    def test_initialization(self):
        """Test that BOCPD initializes correctly."""
        hazard = ConstantHazard(mean_run_length=10.0)
        model = BOCPD(hazard)

        # Check initial state
        self.assertEqual(model.t, 0)
        self.assertEqual(model.R_prev.shape, (model.cfg.max_run_length + 1,))
        self.assertEqual(model.alpha.shape, (model.cfg.max_run_length + 1,))
        self.assertEqual(model.beta.shape, (model.cfg.max_run_length + 1,))

        # Check initial run-length distribution
        self.assertEqual(model.R_prev[0], 1.0)
        self.assertEqual(np.sum(model.R_prev[1:]), 0.0)

        # Check initial parameters
        assert_array_equal(model.alpha, np.full_like(model.alpha, model.cfg.alpha0))
        assert_array_equal(model.beta, np.full_like(model.beta, model.cfg.beta0))

    def test_update_bernoulli(self):
        """Test single update with Bernoulli data."""
        hazard = ConstantHazard(mean_run_length=10.0)
        model = BOCPD(hazard)

        # Update with x_t = 1
        res1 = model.update(1)

        # Check expected values
        self.assertEqual(model.t, 1)
        self.assertEqual(model.alpha[0], model.cfg.alpha0 + 1)
        self.assertEqual(model.beta[0], model.cfg.beta0)

        # Check dictionary result format
        self.assertIn("cp_prob", res1)
        self.assertIn("map_run_length", res1)
        self.assertIn("pred_mean", res1)

        # Update with x_t = 0
        res2 = model.update(0)

        # Check state after second update
        self.assertEqual(model.t, 2)
        self.assertEqual(model.alpha[0], model.cfg.alpha0 + 0)
        self.assertEqual(model.beta[0], model.cfg.beta0 + 1)
        self.assertEqual(model.alpha[1], model.cfg.alpha0 + 1 + 0)
        self.assertEqual(model.beta[1], model.cfg.beta0 + 0 + 1)

    def test_reset(self):
        """Test reset functionality."""
        hazard = ConstantHazard(mean_run_length=10.0)
        model = BOCPD(hazard)

        # Update a few times
        model.update(1)
        model.update(0)
        model.update(1)

        # Reset
        model.reset()

        # Check reset state
        self.assertEqual(model.t, 0)
        self.assertEqual(model.R_prev[0], 1.0)
        self.assertEqual(np.sum(model.R_prev[1:]), 0.0)
        assert_array_equal(model.alpha, np.full_like(model.alpha, model.cfg.alpha0))
        assert_array_equal(model.beta, np.full_like(model.beta, model.cfg.beta0))

    def test_run_batch(self):
        """Test batch processing of a sequence."""
        hazard = ConstantHazard(mean_run_length=10.0)
        cfg = BOCPDConfig(store_run_length_posterior=True)
        model = BOCPD(hazard, cfg)

        # Simple test sequence
        x = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]

        # Run batch process
        result = model.run(x)

        # Check result type and structure
        self.assertIsInstance(result, BOCPDResult)
        self.assertEqual(len(result.cp_prob), len(x))
        self.assertEqual(len(result.map_run_length), len(x))
        self.assertEqual(len(result.pred_mean), len(x))
        self.assertIsNotNone(result.run_length_posterior)
        self.assertEqual(
            result.run_length_posterior.shape, (len(x), model.cfg.max_run_length + 1)
        )

    def test_known_changepoint(self):
        """Test detection of a known changepoint pattern."""
        # Create a sequence with a clear changepoint at t=10
        x = np.concatenate(
            [
                np.zeros(10, dtype=bool),  # p ≈ 0
                np.ones(10, dtype=bool),  # p ≈ 1
            ]
        )

        hazard = ConstantHazard(mean_run_length=20.0)
        cfg = BOCPDConfig(alpha0=1.0, beta0=1.0)
        model = BOCPD(hazard, cfg)

        result = model.run(x)

        # Check that CP probability spikes near the true CP
        cp_idx = np.argmax(result.cp_prob[5:15]) + 5
        self.assertTrue(
            9 <= cp_idx <= 11, f"Detected CP at {cp_idx}, expected around 10"
        )

        # Check that CP probability is high at the detected point
        self.assertTrue(
            result.cp_prob[cp_idx] > 0.5,
            f"CP probability too low: {result.cp_prob[cp_idx]}",
        )


class TestIntegration(unittest.TestCase):
    """Integration tests for BOCPD."""

    def test_complex_pattern(self):
        """Test with a more complex pattern with multiple changepoints."""
        # Pattern with 3 segments
        pattern = """
        0 0 0 0 0 0 0 0 0 0  # First segment (p ≈ 0)
        1 1 1 1 1 1 1 1 1 1  # Second segment (p ≈ 1)
        0 1 0 1 0 1 0 1 0 1  # Third segment (p ≈ 0.5)
        """
        x = parse_binary_string(pattern)

        hazard = ConstantHazard(mean_run_length=15.0)
        model = BOCPD(hazard, BOCPDConfig(store_run_length_posterior=True))

        result = model.run(x)

        # Verify high CP probability near true CPs at t=10 and t=20
        self.assertTrue(
            any(result.cp_prob[9:12] > 0.3),
            f"Missed CP at t≈10: {result.cp_prob[9:12]}",
        )
        self.assertTrue(
            any(result.cp_prob[19:22] > 0.3),
            f"Missed CP at t≈20: {result.cp_prob[19:22]}",
        )

    def test_scheduled_hazard_detection(self):
        """Test that scheduled hazard improves detection at scheduled times."""
        # Create a test sequence where CPs occur at t=10, t=20 (multiples of 10)
        x = np.concatenate(
            [
                np.zeros(10, dtype=bool),  # p ≈ 0
                np.ones(10, dtype=bool),  # p ≈ 1
                np.zeros(10, dtype=bool),  # p ≈ 0
            ]
        )

        # Create scheduled hazard with higher probability at t % 10 == 0
        schedule = [0.5 if i % 10 == 0 else 0.01 for i in range(10)]
        hazard = ScheduledHazard(schedule=schedule, period=10)

        model = BOCPD(hazard, BOCPDConfig(store_run_length_posterior=True))
        result = model.run(x)

        # Expected changepoints
        expected_cps = [10, 20]

        # Check detection at expected CPs
        for cp in expected_cps:
            self.assertTrue(
                result.cp_prob[cp] > 0.5,
                f"Missed scheduled CP at t={cp}: {result.cp_prob[cp]}",
            )


if __name__ == "__main__":
    unittest.main()
