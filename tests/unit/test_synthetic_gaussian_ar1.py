"""Tests for reproducible Gaussian AR(1) changepoint scenarios."""

import numpy as np
import pytest

from changepoint_lab import simulate_gaussian_ar1_segments, simulate_gaussian_segments


def test_zero_autocorrelation_matches_independent_generator() -> None:
    args = ([5, 8], [1.0, -2.0], [0.8, 1.5])
    actual = simulate_gaussian_ar1_segments(*args, autocorrelation=0.0, seed=8)
    baseline = simulate_gaussian_segments(*args, seed=8)
    np.testing.assert_array_equal(actual.values, baseline.values)
    np.testing.assert_array_equal(actual.labels, baseline.labels)
    np.testing.assert_array_equal(actual.changepoints, [5])


def test_seed_and_caller_generator_reproducibility() -> None:
    args = ([20, 15], [0.0, 4.0], [1.0, 2.0])
    first = simulate_gaussian_ar1_segments(*args, autocorrelation=0.75, seed=21)
    second = simulate_gaussian_ar1_segments(*args, autocorrelation=0.75, seed=21)
    np.testing.assert_array_equal(first.values, second.values)
    a = simulate_gaussian_ar1_segments(*args, autocorrelation=-0.3, rng=np.random.default_rng(7))
    b = simulate_gaussian_ar1_segments(*args, autocorrelation=-0.3, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a.values, b.values)
    with pytest.raises(ValueError, match="either seed or rng"):
        simulate_gaussian_ar1_segments(
            *args, autocorrelation=0.2, seed=7, rng=np.random.default_rng(7)
        )


def test_segment_initial_conditions_reset_at_changepoint() -> None:
    args = ([12, 12], [0.0, 100.0], [1.0, 1.0])
    observed = simulate_gaussian_ar1_segments(*args, autocorrelation=0.95, seed=3)
    independent = simulate_gaussian_segments(*args, seed=3)
    assert observed.values[0] == independent.values[0]
    assert observed.values[12] == independent.values[12]
    np.testing.assert_array_equal(observed.changepoints, [12])
    np.testing.assert_array_equal(observed.labels, [0] * 12 + [1] * 12)


def test_constant_segments_remain_constant() -> None:
    sample = simulate_gaussian_ar1_segments(
        [6, 4], [2.0, 8.0], [0.0, 0.0], autocorrelation=-0.8, seed=4
    )
    np.testing.assert_allclose(sample.values, [2.0] * 6 + [8.0] * 4)


def test_empirical_stationary_covariance() -> None:
    """A long segment approximates the specified marginal variance and lag-1 correlation."""
    sample = simulate_gaussian_ar1_segments([100_000], [3.0], [2.0], autocorrelation=0.65, seed=15)
    assert np.mean(sample.values) == pytest.approx(3.0, abs=0.09)
    assert np.std(sample.values) == pytest.approx(2.0, abs=0.09)
    lag_one = np.corrcoef(sample.values[:-1], sample.values[1:])[0, 1]
    assert lag_one == pytest.approx(0.65, abs=0.03)


@pytest.mark.parametrize("phi", [1.0, -1.0, 1.5, float("nan"), float("inf"), "0.5", True])
def test_invalid_autocorrelation(phi: object) -> None:
    with pytest.raises(ValueError, match="autocorrelation"):
        simulate_gaussian_ar1_segments([4], [0.0], [1.0], autocorrelation=phi)  # type: ignore[arg-type]


def test_invalid_segment_contract_is_preserved() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        simulate_gaussian_ar1_segments([0], [0.0], [1.0], autocorrelation=0.3)
