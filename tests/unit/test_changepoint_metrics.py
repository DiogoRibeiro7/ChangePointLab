"""Contract and edge-case tests for tolerance-aware boundary scores."""

import numpy as np
import pytest

from changepoint_lab.core.metrics import score_changepoints


def test_exact_and_tolerant_matches() -> None:
    exact = score_changepoints([10, 30], [10, 31], n_samples=40)
    assert (exact.true_positives, exact.false_positives, exact.false_negatives) == (1, 1, 1)
    assert exact.precision == exact.recall == exact.f1 == 0.5

    tolerant = score_changepoints([10, 30], [10, 31], n_samples=40, tolerance=1)
    assert tolerant.f1 == 1.0
    assert tolerant.matched_pairs == ((10, 10), (30, 31))


def test_duplicate_nearby_predictions_do_not_count_twice() -> None:
    result = score_changepoints([9, 10, 11], [10], n_samples=20, tolerance=2)
    assert result.true_positives == 1
    assert result.false_positives == 2
    assert result.false_negatives == 0
    assert result.precision == pytest.approx(1 / 3)
    assert result.recall == 1.0


def test_sorted_matching_preserves_maximum_cardinality() -> None:
    # Matching 11 to 10 first would wrongly leave 8 unmatched.
    result = score_changepoints([8, 11], [10, 13], n_samples=20, tolerance=2)
    assert result.true_positives == 2
    assert result.matched_pairs == ((8, 10), (11, 13))


@pytest.mark.parametrize(
    ("predicted", "actual", "expected"),
    [([], [], 1.0), ([], [5], 0.0), ([5], [], 0.0)],
)
def test_empty_set_conventions(predicted: list[int], actual: list[int], expected: float) -> None:
    result = score_changepoints(predicted, actual, n_samples=10)
    assert result.precision == result.recall == result.f1 == expected


@pytest.mark.parametrize(
    ("predicted", "actual", "n_samples", "tolerance"),
    [
        ([0], [], 10, 0),
        ([10], [], 10, 0),
        ([4, 4], [], 10, 0),
        ([5, 3], [], 10, 0),
        ([1.5], [], 10, 0),
        ([True], [], 10, 0),
        ([[2]], [], 10, 0),
        ([], [], -1, 0),
        ([], [], 10, -1),
        ([], [], 10, 1.5),
        ([], [], 10, True),
    ],
)
def test_invalid_inputs_are_rejected(predicted, actual, n_samples, tolerance) -> None:
    with pytest.raises(ValueError):
        score_changepoints(predicted, actual, n_samples=n_samples, tolerance=tolerance)


def test_numpy_integer_boundaries() -> None:
    result = score_changepoints(
        np.array([2, 6], dtype=np.int64), [3, 6], n_samples=8, tolerance=np.int64(1)
    )
    assert result.f1 == 1.0
