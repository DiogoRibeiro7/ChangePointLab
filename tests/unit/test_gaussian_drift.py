"""Tests for continuous linear drift applied to labelled Gaussian data."""

import numpy as np
import pytest

from changepoint_lab import (
    GaussianSegmentSample,
    add_gaussian_linear_drift,
    simulate_gaussian_ar1_segments,
    simulate_gaussian_segments,
)


def test_zero_slope_preserves_values_and_ground_truth() -> None:
    baseline = simulate_gaussian_segments([7, 8], [1, 5], [2, 1], seed=3)
    observed = add_gaussian_linear_drift(baseline, slope=0)
    np.testing.assert_array_equal(observed.values, baseline.values)
    np.testing.assert_array_equal(observed.drift, np.zeros(15))
    np.testing.assert_array_equal(observed.changepoints, [7])
    np.testing.assert_array_equal(observed.labels, baseline.labels)
    assert not np.shares_memory(observed.values, baseline.values)


def test_drift_is_continuous_across_boundaries() -> None:
    baseline = simulate_gaussian_segments([3, 4, 2], [0, 0, 0], [0, 0, 0], seed=3)
    observed = add_gaussian_linear_drift(baseline, slope=1.25)
    np.testing.assert_allclose(observed.values, np.arange(9) * 1.25)
    np.testing.assert_allclose(np.diff(observed.drift), np.full(8, 1.25))
    np.testing.assert_array_equal(observed.changepoints, [3, 7])
    np.testing.assert_array_equal(observed.labels, baseline.labels)


def test_negative_slope_composes_with_ar1_and_is_deterministic() -> None:
    baseline = simulate_gaussian_ar1_segments([50, 40], [0, 3], [1, 1], autocorrelation=0.6, seed=4)
    observed = add_gaussian_linear_drift(baseline, slope=-0.02)
    again = add_gaussian_linear_drift(baseline, slope=-0.02)
    np.testing.assert_array_equal(observed.values, again.values)
    np.testing.assert_allclose(observed.values - baseline.values, observed.drift)
    np.testing.assert_array_equal(baseline.changepoints, [50])


@pytest.mark.parametrize("slope", [float("nan"), float("inf"), -float("inf"), True, "0.5"])
def test_invalid_slope(slope: object) -> None:
    sample = simulate_gaussian_segments([5], [0], [1], seed=1)
    with pytest.raises(ValueError, match="slope"):
        add_gaussian_linear_drift(sample, slope=slope)  # type: ignore[arg-type]


def test_nonfinite_result_rejected() -> None:
    sample = simulate_gaussian_segments([4], [0], [0], seed=2)
    with pytest.raises(ValueError, match="nonfinite"):
        add_gaussian_linear_drift(sample, slope=1e308)


def test_rejects_mismatched_inputs() -> None:
    sample = simulate_gaussian_segments([4], [0], [0], seed=2)
    mismatched = GaussianSegmentSample(
        values=sample.values, changepoints=sample.changepoints, labels=np.array([0, 0])
    )
    with pytest.raises(ValueError, match="matching 1-D"):
        add_gaussian_linear_drift(mismatched, slope=0.5)
