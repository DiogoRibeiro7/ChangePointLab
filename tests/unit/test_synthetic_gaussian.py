"""Unit tests for labelled synthetic changepoint datasets."""

import numpy as np
import pytest

from changepoint_lab import simulate_gaussian_segments
from changepoint_lab.core.segmentation import changepoints_from_labels


def test_reproducibility_and_right_exclusive_boundaries() -> None:
    kwargs = {"lengths": [4, 3, 5], "means": [0, 2, -1], "standard_deviations": [1, 2, 1]}
    first = simulate_gaussian_segments(**kwargs, seed=42)
    second = simulate_gaussian_segments(**kwargs, seed=42)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.changepoints, [4, 7])
    np.testing.assert_array_equal(first.labels, [0] * 4 + [1] * 3 + [2] * 5)
    np.testing.assert_array_equal(changepoints_from_labels(first.labels), first.changepoints)


def test_single_segment_and_deterministic_shifts() -> None:
    sample = simulate_gaussian_segments([2, 3], [1.0, 5.0], [0.0, 0.0], seed=0)
    np.testing.assert_array_equal(sample.values, [1, 1, 5, 5, 5])
    np.testing.assert_array_equal(sample.changepoints, [2])

    single = simulate_gaussian_segments([4], [2.0], [0.0], seed=0)
    assert single.changepoints.size == 0
    assert single.labels.tolist() == [0, 0, 0, 0]


def test_caller_generator_reproducibility() -> None:
    args = ([10], [0.0], [1.0])
    a = simulate_gaussian_segments(*args, rng=np.random.default_rng(4))
    b = simulate_gaussian_segments(*args, rng=np.random.default_rng(4))
    np.testing.assert_array_equal(a.values, b.values)
    with pytest.raises(ValueError, match="either seed or rng"):
        simulate_gaussian_segments(*args, seed=4, rng=np.random.default_rng(4))


@pytest.mark.parametrize(
    ("lengths", "means", "stds"),
    [
        ([], [], []),
        ([0], [0], [1]),
        ([-1], [0], [1]),
        ([1.5], [0], [1]),
        ([True], [0], [1]),
        ([1, 2], [0], [1, 1]),
        ([1], [float("nan")], [1]),
        ([1], [0], [float("inf")]),
        ([1], [0], [-1]),
        ([[2]], [0], [1]),
    ],
)
def test_invalid_parameters_raise(lengths, means, stds) -> None:
    with pytest.raises(ValueError):
        simulate_gaussian_segments(lengths, means, stds, seed=1)
