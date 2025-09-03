# test_bayesian_blocks_fixed.py
# Fixed test file with correct imports and working functionality

from __future__ import annotations

import numpy as np
import pytest
import time
import warnings
from typing import List, Tuple, Callable

# FIXED IMPORTS - using the correct module names
from bayesian_blocks import (
    bayesian_blocks_events,
    bayesian_blocks_counts,
    bayesian_blocks_bernoulli,
    bayesian_blocks,  # unified API
    BBResult,
    BBConfig,
    DataType,
    _detect_data_type,
    ncp_prior_from_p0,
)

# Import utilities with correct module name
from bayesian_blocks.advanced_utils import (
    bootstrap_confidence_intervals,
    cross_validate_parameters,
    StreamingBayesianBlocks,
    AdaptiveBayesianBlocks,
    detect_outlier_blocks,
    quick_analysis,
)


class TestBasicFunctionality:
    """Test core functionality with various data types."""

    def test_empty_data(self):
        """Test behavior with empty inputs."""
        # Empty arrays should return empty results
        result = bayesian_blocks_counts([])
        assert len(result.edges) == 0
        assert len(result.block_value) == 0
        assert len(result.change_points) == 0

        # Events with empty array
        result = bayesian_blocks_events([])
        assert len(result.edges) == 0

    def test_single_point_data(self):
        """Test with single data point."""
        # Single count
        result = bayesian_blocks_counts([5.0])
        assert len(result.block_value) == 1
        assert result.block_value[0] == 5.0

        # Single event
        result = bayesian_blocks_events([1.0])
        assert len(result.block_value) == 1

        # Single binary
        result = bayesian_blocks_bernoulli([1])
        assert len(result.block_value) == 1
        assert result.block_value[0] == 1.0

    def test_identical_data(self):
        """Test with identical values."""
        n = 100
        constant_value = 3.5

        # Identical counts should yield single block
        result = bayesian_blocks_counts([constant_value] * n, p0=0.05)
        assert len(result.block_value) == 1
        assert abs(result.block_value[0] - constant_value) < 1e-10

        # Identical binary values
        result = bayesian_blocks_bernoulli([1] * n, p0=0.05)
        assert len(result.block_value) == 1
        assert result.block_value[0] == 1.0

    def test_extreme_penalty_values(self):
        """Test with extreme p0/gamma values."""
        rng = np.random.default_rng(42)
        data = rng.poisson(2.0, 200)

        # Very small p0 should give many blocks
        result_small_p0 = bayesian_blocks_counts(data, p0=1e-6)

        # Very large p0 should give few blocks
        result_large_p0 = bayesian_blocks_counts(data, p0=0.99)

        assert len(result_small_p0.block_value) > len(result_large_p0.block_value)

        # Direct penalty testing
        result_small_pen = bayesian_blocks_counts(data, penalty=0.1)
        result_large_pen = bayesian_blocks_counts(data, penalty=100.0)

        assert len(result_small_pen.block_value) > len(result_large_pen.block_value)

    def test_invalid_inputs(self):
        """Test various invalid inputs."""
        # Negative counts
        with pytest.raises(ValueError, match="must be >= 0"):
            bayesian_blocks_counts([-1, 2, 3])

        # Invalid p0
        with pytest.raises(ValueError, match="p0 must be in"):
            bayesian_blocks_counts([1, 2, 3], p0=1.5)

        with pytest.raises(ValueError, match="p0 must be in"):
            bayesian_blocks_counts([1, 2, 3], p0=0.0)

        # Successes > trials
        with pytest.raises(ValueError, match="successes must be <= trials"):
            bayesian_blocks_bernoulli([3], [2])

        # Negative widths - this should be caught by validation
        with pytest.raises(ValueError):
            bayesian_blocks_counts([1, 2], [-1, 1])

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very large values
        large_data = [1e10, 1e10 + 1, 1e10 + 2]
        result = bayesian_blocks_counts(large_data)
        assert np.isfinite(result.block_value).all()

        # Very small positive values
        small_data = [1e-10, 2e-10, 3e-10]
        result = bayesian_blocks_counts(small_data)
        assert np.isfinite(result.block_value).all()

        # Mixed scales
        mixed_data = [1e-6, 1e6, 1e-6, 1e6]
        result = bayesian_blocks_counts(mixed_data)
        assert np.isfinite(result.block_value).all()


class TestUnifiedAPI:
    """Test the new unified API."""

    def test_unified_api_counts(self):
        """Test unified API with count data."""
        rng = np.random.default_rng(42)
        count_data = rng.poisson(2.0, 100)

        # Test explicit type
        result1 = bayesian_blocks(count_data, data_type="counts")
        assert result1 is not None
        assert len(result1.block_value) > 0

        # Test auto-detection
        result2 = bayesian_blocks(count_data, data_type="auto")
        assert result2 is not None
        assert len(result2.block_value) > 0

        # Should be same results (approximately)
        assert len(result1.block_value) == len(result2.block_value)

    def test_unified_api_events(self):
        """Test unified API with event data."""
        rng = np.random.default_rng(42)
        event_data = np.cumsum(rng.exponential(0.5, 50))

        result = bayesian_blocks(event_data, data_type="events")
        assert result is not None
        assert len(result.block_value) > 0

    def test_unified_api_bernoulli(self):
        """Test unified API with binary data."""
        rng = np.random.default_rng(42)
        binary_data = rng.binomial(1, 0.3, 100)

        result = bayesian_blocks(binary_data, data_type="bernoulli")
        assert result is not None
        assert len(result.block_value) > 0
        assert np.all(result.block_value >= 0)
        assert np.all(result.block_value <= 1)

    def test_config_object(self):
        """Test using BBConfig object."""
        rng = np.random.default_rng(42)
        data = rng.poisson(2.0, 100)

        config = BBConfig(p0=0.01, min_block_size=2)
        result = bayesian_blocks(data, data_type="counts", config=config)

        assert result is not None
        assert result.config == config


class TestDataTypeDetection:
    """Test automatic data type detection."""

    def test_detect_binary_data(self):
        """Test detection of binary sequences."""
        binary_data = [0, 1, 1, 0, 1]
        detected = _detect_data_type(binary_data)
        assert detected == DataType.BERNOULLI

    def test_detect_count_data(self):
        """Test detection of count data."""
        count_data = [0, 2, 5, 1, 8]  # Non-negative integers
        detected = _detect_data_type(count_data)
        assert detected == DataType.COUNTS

    def test_detect_event_data(self):
        """Test detection of continuous event times."""
        event_data = [1.2, 2.7, 3.1, 5.9]  # Continuous values
        detected = _detect_data_type(event_data)
        assert detected == DataType.EVENTS

    def test_detect_tuple_data(self):
        """Test detection with tuple inputs."""
        # (successes, trials)
        bernoulli_tuple = ([1, 0, 1, 1], [1, 1, 1, 1])
        detected = _detect_data_type(bernoulli_tuple)
        assert detected == DataType.BERNOULLI

        # (counts, widths)
        counts_tuple = ([3, 5, 2], [1.0, 1.5, 0.8])
        detected = _detect_data_type(counts_tuple)
        assert detected == DataType.COUNTS


class TestStatisticalProperties:
    """Test statistical properties and correctness."""

    def test_known_changepoint_detection(self):
        """Test detection of known changepoints."""
        rng = np.random.default_rng(42)

        # Create data with known changepoint at position 100
        n1, n2 = 100, 100
        rate1, rate2 = 5.0, 1.0

        data1 = rng.poisson(rate1, n1)
        data2 = rng.poisson(rate2, n2)
        data = np.concatenate([data1, data2])

        result = bayesian_blocks_counts(data, p0=0.05)

        # Should detect at least one changepoint
        assert len(result.block_value) >= 2

        # First block should have higher rate than last
        assert result.block_value[0] > result.block_value[-1]

        # Changepoint should be roughly near position 100
        if len(result.change_points) > 0:
            closest_cp = min(result.change_points, key=lambda x: abs(x - 100))
            assert abs(closest_cp - 100) < 30  # Allow some tolerance

    def test_no_false_positives_constant(self):
        """Test that constant data doesn't produce false positives."""
        rng = np.random.default_rng(42)

        # Generate from single Poisson distribution
        n = 300
        true_rate = 3.0
        data = rng.poisson(true_rate, n)

        # With strong penalty, should prefer single block
        result = bayesian_blocks_counts(data, p0=0.001)

        # Should detect only one block most of the time
        assert len(result.block_value) <= 3  # Allow some occasional splits

        # Rate should be close to true rate
        overall_rate = np.mean(data)
        assert abs(result.block_value[0] - overall_rate) < 1.0

    def test_bernoulli_probability_estimation(self):
        """Test Bernoulli probability estimation accuracy."""
        rng = np.random.default_rng(42)

        # Known probability change
        n1, n2 = 150, 150
        p1, p2 = 0.2, 0.8

        x1 = rng.binomial(1, p1, n1)
        x2 = rng.binomial(1, p2, n2)
        x = np.concatenate([x1, x2])

        result = bayesian_blocks_bernoulli(x, p0=0.05)

        # Should detect change
        assert len(result.block_value) >= 2

        # First block probability should be close to p1
        assert abs(result.block_value[0] - p1) < 0.15

        # Last block probability should be close to p2
        assert abs(result.block_value[-1] - p2) < 0.15

    def test_event_rate_estimation(self):
        """Test event rate estimation for Poisson processes."""
        rng = np.random.default_rng(42)

        # Generate two-rate Poisson process
        rate1, rate2 = 3.0, 0.5
        t_change = 5.0
        t_end = 10.0

        # Generate event times
        t1 = np.cumsum(rng.exponential(1 / rate1, 200))
        t1 = t1[t1 < t_change]

        t2 = t_change + np.cumsum(rng.exponential(1 / rate2, 200))
        t2 = t2[t2 < t_end]

        events = np.sort(np.concatenate([t1, t2]))

        result = bayesian_blocks_events(events, t_start=0.0, t_stop=t_end, penalty=0.5)

        # Should detect rate change
        assert len(result.block_value) >= 2

        # First rate should be higher
        assert result.block_value[0] > result.block_value[-1]


class TestBackwardCompatibility:
    """Test that original API still works."""

    def test_original_counts_api(self):
        """Test original bayesian_blocks_counts API."""
        rng = np.random.default_rng(42)
        data = rng.poisson(2.0, 100)

        # Original API should work
        result = bayesian_blocks_counts(data, p0=0.05)
        assert result is not None
        assert len(result.block_value) > 0

        # With widths
        widths = np.ones(len(data)) * 1.5
        result2 = bayesian_blocks_counts(data, widths, p0=0.05)
        assert result2 is not None

    def test_original_events_api(self):
        """Test original bayesian_blocks_events API."""
        rng = np.random.default_rng(42)
        events = np.cumsum(rng.exponential(0.5, 50))

        result = bayesian_blocks_events(events, p0=0.05)
        assert result is not None
        assert len(result.block_value) > 0

        # With time bounds
        result2 = bayesian_blocks_events(
            events, t_start=0.0, t_stop=events[-1], p0=0.05
        )
        assert result2 is not None

    def test_original_bernoulli_api(self):
        """Test original bayesian_blocks_bernoulli API."""
        rng = np.random.default_rng(42)
        binary = rng.binomial(1, 0.3, 100)

        result = bayesian_blocks_bernoulli(binary, p0=0.05)
        assert result is not None
        assert len(result.block_value) > 0
        assert np.all(result.block_value >= 0)
        assert np.all(result.block_value <= 1)


class TestEnhancedFeatures:
    """Test enhanced BBResult features."""

    def test_enhanced_result_properties(self):
        """Test that BBResult has enhanced properties."""
        rng = np.random.default_rng(42)
        data = rng.poisson(2.0, 100)

        result = bayesian_blocks_counts(data, p0=0.05)

        # Should have enhanced properties
        assert hasattr(result, "n_blocks")
        assert hasattr(result, "aic")
        assert hasattr(result, "bic")
        assert hasattr(result, "log_likelihood")
        assert hasattr(result, "config")

        # Values should be reasonable
        assert result.n_blocks == len(result.block_value)
        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)
        assert np.isfinite(result.log_likelihood)


class TestRobustness:
    """Test robustness to various edge cases."""

    def test_nan_handling(self):
        """Test handling of NaN values."""
        data_with_nan = [1.0, 2.0, np.nan, 3.0, 4.0]

        with pytest.raises(ValueError, match="non-finite"):
            bayesian_blocks_counts(data_with_nan)

    def test_inf_handling(self):
        """Test handling of infinite values."""
        data_with_inf = [1.0, 2.0, np.inf, 3.0, 4.0]

        with pytest.raises(ValueError, match="non-finite"):
            bayesian_blocks_counts(data_with_inf)

    def test_very_small_datasets(self):
        """Test with very small datasets."""
        # Single point
        result = bayesian_blocks_counts([5.0])
        assert len(result.block_value) == 1

        # Two points
        result = bayesian_blocks_counts([3.0, 7.0])
        assert len(result.block_value) >= 1


# Quick smoke test that can be run directly
if __name__ == "__main__":
    print("Running smoke tests...")

    # Basic functionality
    rng = np.random.default_rng(42)

    # Test counts
    count_data = rng.poisson(2.0, 100)
    result = bayesian_blocks_counts(count_data)
    print(f"Counts: {len(result.block_value)} blocks detected")

    # Test events
    event_data = np.cumsum(rng.exponential(0.5, 50))
    result = bayesian_blocks_events(event_data)
    print(f"Events: {len(result.block_value)} blocks detected")

    # Test Bernoulli
    binary_data = rng.binomial(1, 0.3, 100)
    result = bayesian_blocks_bernoulli(binary_data)
    print(f"Bernoulli: {len(result.block_value)} blocks detected")

    # Test unified API
    result = bayesian_blocks(count_data, data_type="auto")
    print(f"Unified API: {len(result.block_value)} blocks detected")

    print("All smoke tests passed!")
    print(
        "Run with pytest for full test suite: pytest test_bayesian_blocks_fixed.py -v"
    )
