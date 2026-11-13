"""Evaluation of right-exclusive changepoint boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .segmentation import normalize_linear_changepoints


@dataclass(frozen=True)
class ChangePointScores:
    """One-to-one tolerance-based detection scores.

    When both sets are empty, precision, recall and F1 are defined as 1.0.
    A nonempty set compared with an empty set receives zero for all scores.
    """

    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_pairs: tuple[tuple[int, int], ...]


def _validated_boundaries(indices: Sequence[int] | NDArray[np.integer], n: int) -> list[int]:
    """Reject fractional, unordered and out-of-range boundary indices."""
    values = np.asarray(indices)
    if values.ndim != 1 or (values.size > 0 and values.dtype.kind not in "iu"):
        raise ValueError("changepoints must be a 1-D sequence of integers")
    # Do not allow normalize_linear_changepoints to silently truncate floats.
    return normalize_linear_changepoints(values, n=n).tolist()


def score_changepoints(
    predicted: Sequence[int] | NDArray[np.integer],
    actual: Sequence[int] | NDArray[np.integer],
    *,
    n_samples: int,
    tolerance: int = 0,
) -> ChangePointScores:
    """Score detected linear changepoints against ground truth.

    All boundaries are right-exclusive indices in ``1..n_samples-1``.
    Each predicted boundary can match at most one actual boundary within
    ``tolerance`` observations. Sorted two-pointer matching maximises the
    number of matches; it does not minimise total matching distance.

    Parameters
    ----------
    predicted, actual:
        Strictly increasing, duplicate-free integer boundary sequences.
    n_samples:
        Number of observations in the original series.
    tolerance:
        Maximum absolute index difference for a match, in observations.

    Returns
    -------
    ChangePointScores
        Precision, recall, F1 and matched ``(predicted, actual)`` pairs.
    """
    if not isinstance(n_samples, (int, np.integer)) or isinstance(n_samples, bool):
        raise ValueError("n_samples must be a non-negative integer")
    if not isinstance(tolerance, (int, np.integer)) or isinstance(tolerance, bool):
        raise ValueError("tolerance must be a non-negative integer")
    if n_samples < 0 or tolerance < 0:
        raise ValueError("n_samples and tolerance must be non-negative")

    preds = _validated_boundaries(predicted, int(n_samples))
    truths = _validated_boundaries(actual, int(n_samples))
    i = j = 0
    pairs: list[tuple[int, int]] = []
    while i < len(preds) and j < len(truths):
        if preds[i] < truths[j] - tolerance:
            i += 1
        elif truths[j] < preds[i] - tolerance:
            j += 1
        else:
            pairs.append((preds[i], truths[j]))
            i += 1
            j += 1

    tp = len(pairs)
    fp = len(preds) - tp
    fn = len(truths) - tp
    if not preds and not truths:
        precision = recall = 1.0
    else:
        precision = tp / len(preds) if preds else 0.0
        recall = tp / len(truths) if truths else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    return ChangePointScores(precision, recall, f1, tp, fp, fn, tuple(pairs))
