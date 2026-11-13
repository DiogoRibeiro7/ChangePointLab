"""Tests for comparable offline detector outputs and evaluation."""

import numpy as np
import pytest

from changepoint_lab import ChangePointResult, compare_detectors, simulate_gaussian_segments
from changepoint_lab.algorithms.optimization.pelt import NormalMeanKnownVar
from changepoint_lab import PELT


class FixedDetector:
    """Minimal detector stub with an observable fit call."""

    def __init__(self, boundaries: list[int], convention: str = "right_exclusive") -> None:
        self.boundaries = boundaries
        self.convention = convention
        self.seen: np.ndarray | None = None

    def fit_predict(self, x: np.ndarray) -> ChangePointResult:
        self.seen = x.copy()
        return ChangePointResult(
            indices=np.asarray(self.boundaries), boundary_convention=self.convention
        )


def test_compare_named_detectors_with_ground_truth() -> None:
    sample = simulate_gaussian_segments([4, 4], [0, 5], [0, 0], seed=1)
    comparisons = compare_detectors(
        {"exact": FixedDetector([4]), "near": FixedDetector([5])},
        sample.values,
        actual=sample.changepoints,
        tolerance=1,
    )
    assert [entry.name for entry in comparisons] == ["exact", "near"]
    assert all(entry.scores is not None and entry.scores.f1 == 1.0 for entry in comparisons)


def test_no_reference_provides_raw_results() -> None:
    (entry,) = compare_detectors({"baseline": FixedDetector([])}, [0.0, 1.0, 2.0])
    assert entry.scores is None
    assert entry.result.indices.size == 0


def test_shared_input_is_not_mutated() -> None:
    class MutatingDetector(FixedDetector):
        def fit_predict(self, x: np.ndarray) -> ChangePointResult:
            x[:] = -100
            return super().fit_predict(x)

    original = np.asarray([1.0, 2.0, 3.0])
    observer = FixedDetector([])
    compare_detectors({"mutates": MutatingDetector([]), "observes": observer}, original)
    np.testing.assert_array_equal(original, [1, 2, 3])
    np.testing.assert_array_equal(observer.seen, [1, 2, 3])


def test_integration_with_pelt() -> None:
    sample = simulate_gaussian_segments([5, 5], [0.0, 5.0], [0.0, 0.0], seed=3)
    (entry,) = compare_detectors(
        {"pelt": PELT(NormalMeanKnownVar(sigma2=1.0), penalty=1.0, min_seg_len=2)},
        sample.values,
        actual=sample.changepoints,
    )
    assert entry.scores is not None and entry.scores.f1 == 1.0


@pytest.mark.parametrize(
    ("data", "actual", "tolerance"),
    [([1], None, 0), ([float("nan"), 1], None, 0), ([1, 2], [0], 0), ([1, 2], None, 1)],
)
def test_invalid_inputs(data, actual, tolerance) -> None:
    with pytest.raises(ValueError):
        compare_detectors({"test": FixedDetector([])}, data, actual=actual, tolerance=tolerance)


def test_wrong_boundary_convention_rejected() -> None:
    with pytest.raises(ValueError, match="time_index"):
        compare_detectors({"online": FixedDetector([1], "time_index")}, [1, 2, 3])


def test_detector_exception_is_named() -> None:
    class BrokenDetector:
        def fit_predict(self, x: np.ndarray) -> ChangePointResult:
            raise ArithmeticError("failed")

    with pytest.raises(RuntimeError, match="broken"):
        compare_detectors({"broken": BrokenDetector()}, [1.0, 2.0])
