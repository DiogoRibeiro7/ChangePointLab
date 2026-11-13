"""Compare offline changepoint detectors under shared input and scoring rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .datatypes import ChangePointResult
from .metrics import ChangePointScores, score_changepoints
from .segmentation import normalize_linear_changepoints


class ComparableDetector(Protocol):
    """Minimal contract for a compatible offline detector."""

    def fit_predict(self, x: np.ndarray) -> ChangePointResult:
        """Fit on observations and return a typed changepoint result."""
        ...


@dataclass(frozen=True)
class DetectorComparison:
    """Output from one detector, with optional reference-based scores."""

    name: str
    result: ChangePointResult
    scores: ChangePointScores | None


def compare_detectors(
    detectors: Mapping[str, ComparableDetector],
    data: NDArray[np.floating] | Sequence[float],
    *,
    actual: Sequence[int] | NDArray[np.integer] | None = None,
    tolerance: int = 0,
) -> tuple[DetectorComparison, ...]:
    """Fit detectors on the same observations and optionally evaluate detections.

    Only results using right-exclusive interior boundary indices are accepted.
    A separate copy of the observations is provided to each detector, avoiding
    mutation of shared input. Detectors are run in mapping insertion order.

    Existing detector instances are fitted in place; create fresh instances
    for independent benchmark repetitions. This function does not compare
    algorithm-specific objective scores, which are generally incommensurable.
    Errors from individual detectors propagate with the detector name attached.
    """
    if not detectors:
        raise ValueError("detectors must contain at least one named detector")
    values = np.asarray(data)
    if values.ndim not in (1, 2) or values.shape[0] < 2:
        raise ValueError("data must be a 1-D or 2-D array with at least two observations")
    if values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)):
        raise ValueError("data must contain finite numeric observations")
    if actual is None and tolerance != 0:
        raise ValueError("tolerance requires actual reference changepoints")

    n_samples = int(values.shape[0])
    if actual is not None:
        # Validate reference boundaries before fitting any detector.
        score_changepoints([], actual, n_samples=n_samples, tolerance=tolerance)

    comparisons: list[DetectorComparison] = []
    for name, detector in detectors.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("detector names must be nonempty strings")
        try:
            result = detector.fit_predict(np.array(values, copy=True))
        except Exception as exc:
            raise RuntimeError(f"detector {name!r} failed") from exc
        if not isinstance(result, ChangePointResult):
            raise TypeError(f"detector {name!r} did not return ChangePointResult")
        if result.boundary_convention != "right_exclusive":
            raise ValueError(f"detector {name!r} uses {result.boundary_convention!r} boundaries")
        normalize_linear_changepoints(result.indices, n=n_samples)
        scores = (
            score_changepoints(result.indices, actual, n_samples=n_samples, tolerance=tolerance)
            if actual is not None
            else None
        )
        comparisons.append(DetectorComparison(name=name, result=result, scores=scores))
    return tuple(comparisons)
