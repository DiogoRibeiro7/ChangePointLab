"""Reproducible synthetic data with known linear changepoint boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .random import make_rng


@dataclass(frozen=True)
class GaussianSegmentSample:
    """Synthetic observations and their right-exclusive reference boundaries.

    ``values`` and ``labels`` are indexed by observation; ``changepoints``
    contains the first index of every segment after the first.
    """

    values: NDArray[np.float64]
    changepoints: NDArray[np.int_]
    labels: NDArray[np.int_]


def simulate_gaussian_segments(
    lengths: Sequence[int],
    means: Sequence[float],
    standard_deviations: Sequence[float],
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> GaussianSegmentSample:
    """Sample independent Gaussian segments with exactly known boundaries.

    Each segment may change its mean, variance, or both. A standard deviation
    of zero produces a deterministic constant segment. Segment lengths must be
    positive integers; means must be finite and deviations finite/nonnegative.

    Provide ``seed`` for reproducible samples or ``rng`` to use a caller-owned
    stream, but not both. The returned boundaries use the package's linear
    right-exclusive convention and exclude the final observation count.
    """
    segment_lengths = np.asarray(lengths)
    segment_means = np.asarray(means, dtype=float)
    segment_stds = np.asarray(standard_deviations, dtype=float)
    if segment_lengths.ndim != 1 or segment_lengths.size == 0:
        raise ValueError("lengths must be a nonempty 1-D sequence")
    if segment_lengths.dtype.kind not in "iu" or np.any(segment_lengths <= 0):
        raise ValueError("segment lengths must be positive integers")
    if segment_means.ndim != 1 or segment_stds.ndim != 1:
        raise ValueError("means and standard_deviations must be 1-D sequences")
    if not (len(segment_lengths) == len(segment_means) == len(segment_stds)):
        raise ValueError("lengths, means and standard_deviations must have equal lengths")
    if not np.all(np.isfinite(segment_means)):
        raise ValueError("means must be finite")
    if not np.all(np.isfinite(segment_stds)) or np.any(segment_stds < 0):
        raise ValueError("standard deviations must be finite and nonnegative")
    if segment_lengths.sum(dtype=np.int64) > np.iinfo(np.intp).max:
        raise ValueError("total length exceeds supported array size")

    generator = make_rng(seed=seed, rng=rng)
    # The first segment begins at zero; the final cumulative edge is n_samples.
    edges = np.cumsum(segment_lengths, dtype=np.intp)
    values = np.concatenate(
        [
            generator.normal(float(mean), float(std), size=int(length))
            for length, mean, std in zip(segment_lengths, segment_means, segment_stds, strict=True)
        ]
    ).astype(np.float64, copy=False)
    labels = np.repeat(np.arange(len(segment_lengths), dtype=int), segment_lengths)
    return GaussianSegmentSample(
        values=values,
        changepoints=edges[:-1].astype(int, copy=False),
        labels=labels.astype(int, copy=False),
    )
