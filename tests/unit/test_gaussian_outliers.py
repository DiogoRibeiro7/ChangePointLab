"""Tests for composable outlier contamination of Gaussian changepoint data."""

import numpy as np
import pytest

from changepoint_lab import (
    contaminate_gaussian_sample,
    simulate_gaussian_ar1_segments,
    simulate_gaussian_segments,
)


def test_zero_probability_preserves_values_and_boundaries() -> None:
    original = simulate_gaussian_segments([10, 12], [0, 3], [1, 1], seed=4)
    contaminated = contaminate_gaussian_sample(
        original, probability=0, outlier_standard_deviation=8, seed=5
    )
    assert not np.any(contaminated.outlier_mask)
    np.testing.assert_array_equal(original.values, contaminated.values)
    np.testing.assert_array_equal(original.labels, contaminated.labels)
    np.testing.assert_array_equal(original.changepoints, contaminated.changepoints)
    assert not np.shares_memory(original.values, contaminated.values)


def test_all_observations_contaminated_without_modifying_input() -> None:
    original = simulate_gaussian_segments([6, 7], [0, 2], [0, 0], seed=1)
    untouched = original.values.copy()
    first = contaminate_gaussian_sample(
        original, probability=1, outlier_standard_deviation=5, seed=3
    )
    second = contaminate_gaussian_sample(
        original, probability=1, outlier_standard_deviation=5, seed=3
    )
    assert np.all(first.outlier_mask)
    np.testing.assert_array_equal(first.values, second.values)
    assert np.any(first.values != original.values)
    np.testing.assert_array_equal(original.values, untouched)
    np.testing.assert_array_equal(first.changepoints, [6])
    np.testing.assert_array_equal(first.labels, [0] * 6 + [1] * 7)


def test_seeded_contamination_is_composable_with_ar1() -> None:
    baseline = simulate_gaussian_ar1_segments(
        [400, 300], [0, 4], [1, 2], autocorrelation=0.6, seed=8
    )
    a = contaminate_gaussian_sample(
        baseline, probability=0.2, outlier_standard_deviation=20, rng=np.random.default_rng(9)
    )
    b = contaminate_gaussian_sample(
        baseline, probability=0.2, outlier_standard_deviation=20, rng=np.random.default_rng(9)
    )
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.outlier_mask, b.outlier_mask)
    assert 100 < a.outlier_mask.sum() < 180
    assert np.all(a.values[~a.outlier_mask] == baseline.values[~a.outlier_mask])
    np.testing.assert_array_equal(a.changepoints, [400])


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_probability(probability: object) -> None:
    baseline = simulate_gaussian_segments([4], [0], [1], seed=1)
    with pytest.raises(ValueError, match="probability"):
        contaminate_gaussian_sample(  # type: ignore[arg-type]
            baseline, probability=probability, outlier_standard_deviation=5
        )


@pytest.mark.parametrize("scale", [-1.0, 0.0, float("nan"), float("inf"), True])
def test_invalid_outlier_scale(scale: object) -> None:
    baseline = simulate_gaussian_segments([4], [0], [1], seed=1)
    with pytest.raises(ValueError, match="outlier_standard_deviation"):
        contaminate_gaussian_sample(  # type: ignore[arg-type]
            baseline, probability=0.2, outlier_standard_deviation=scale
        )


def test_seed_and_rng_cannot_be_combined() -> None:
    baseline = simulate_gaussian_segments([4], [0], [1], seed=1)
    with pytest.raises(ValueError, match="either seed or rng"):
        contaminate_gaussian_sample(
            baseline,
            probability=0.2,
            outlier_standard_deviation=2,
            seed=2,
            rng=np.random.default_rng(2),
        )
