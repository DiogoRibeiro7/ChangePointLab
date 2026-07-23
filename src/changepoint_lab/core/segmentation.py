from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

ArrayI: TypeAlias = NDArray[np.int_]


def normalize_linear_changepoints(
    indices: NDArray[np.integer] | list[int] | tuple[int, ...],
    *,
    n: int,
    min_segment_length: int = 1,
) -> ArrayI:
    """Validate linear changepoints as right-exclusive interior boundaries.

    A changepoint ``k`` means that the previous segment is ``[..., k)`` and the
    next segment starts at observation ``k``. Valid changepoints are sorted,
    duplicate-free integers in ``1..n-1``.
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    if min_segment_length < 1:
        raise ValueError("min_segment_length must be positive.")
    cps = np.asarray(indices, dtype=int)
    if cps.ndim != 1:
        raise ValueError("changepoints must be a 1-D sequence.")
    if cps.size == 0:
        if n > 0 and n < min_segment_length:
            raise ValueError("single segment is shorter than min_segment_length.")
        return cps
    if n < 2:
        raise ValueError("length-one linear data cannot contain changepoints.")
    if np.any(cps <= 0) or np.any(cps >= n):
        raise ValueError("linear changepoints must be interior boundaries in 1..n-1.")
    if np.any(np.diff(cps) <= 0):
        raise ValueError("linear changepoints must be strictly increasing.")
    edges = np.concatenate(([0], cps, [n]))
    if np.any(np.diff(edges) < min_segment_length):
        raise ValueError("at least one segment is shorter than min_segment_length.")
    return cps.astype(int, copy=False)


def changepoints_to_edges(
    indices: NDArray[np.integer] | list[int] | tuple[int, ...],
    *,
    n: int,
    min_segment_length: int = 1,
) -> ArrayI:
    """Convert right-exclusive changepoints to half-open segment edges."""
    cps = normalize_linear_changepoints(
        indices,
        n=n,
        min_segment_length=min_segment_length,
    )
    return np.concatenate(([0], cps, [n])).astype(int, copy=False)


def edges_to_changepoints(
    edges: NDArray[np.integer] | list[int] | tuple[int, ...],
    *,
    n: int | None = None,
    min_segment_length: int = 1,
) -> ArrayI:
    """Convert half-open segment edges ``[0, *cps, n]`` to changepoints."""
    edge_arr = np.asarray(edges, dtype=int)
    if edge_arr.ndim != 1 or edge_arr.size < 2:
        raise ValueError("edges must be a 1-D sequence with at least start and stop.")
    if edge_arr[0] != 0:
        raise ValueError("linear segment edges must start at 0.")
    inferred_n = int(edge_arr[-1])
    if n is not None and inferred_n != n:
        raise ValueError("last edge must equal n.")
    if inferred_n < 0:
        raise ValueError("last edge must be non-negative.")
    if np.any(np.diff(edge_arr) < min_segment_length):
        raise ValueError("segment edges violate min_segment_length.")
    cps = edge_arr[1:-1]
    return normalize_linear_changepoints(
        cps,
        n=inferred_n,
        min_segment_length=min_segment_length,
    )


def labels_from_changepoints(
    n: int,
    indices: NDArray[np.integer] | list[int] | tuple[int, ...],
    *,
    min_segment_length: int = 1,
) -> ArrayI:
    """Build contiguous segment labels from right-exclusive changepoints."""
    edges = changepoints_to_edges(
        indices,
        n=n,
        min_segment_length=min_segment_length,
    )
    labels = np.empty(n, dtype=int)
    for label, (start, stop) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        labels[int(start) : int(stop)] = label
    return labels


def changepoints_from_labels(labels: NDArray[np.integer] | list[int]) -> ArrayI:
    """Return right-exclusive boundaries where adjacent labels differ."""
    label_arr = np.asarray(labels, dtype=int)
    if label_arr.ndim != 1:
        raise ValueError("labels must be a 1-D sequence.")
    if label_arr.size < 2:
        return np.array([], dtype=int)
    return (np.flatnonzero(np.diff(label_arr) != 0) + 1).astype(int, copy=False)


def segment_slices(
    n: int,
    indices: NDArray[np.integer] | list[int] | tuple[int, ...],
    *,
    min_segment_length: int = 1,
) -> tuple[slice, ...]:
    """Return Python half-open slices for each linear segment."""
    edges = changepoints_to_edges(
        indices,
        n=n,
        min_segment_length=min_segment_length,
    )
    return tuple(slice(int(start), int(stop)) for start, stop in zip(edges[:-1], edges[1:], strict=True))


@dataclass(frozen=True)
class CircularSegment:
    """A segment on a periodic lattice, represented by inclusive bin endpoints."""

    start: int
    end: int
    length: int


@dataclass(frozen=True)
class CircularChangePoints:
    """Explicit circular changepoints using periodic bin-end semantics.

    Each changepoint ``k`` is the last bin of a segment on ``0..period-1``. The
    next segment starts at ``(k + 1) % period``.
    """

    period: int
    indices: ArrayI
    boundary_convention: str = "periodic_bin_end"

    def __post_init__(self) -> None:
        if self.period < 1:
            raise ValueError("period must be positive.")
        indices = np.asarray(self.indices, dtype=int)
        if indices.ndim != 1:
            raise ValueError("circular changepoints must be a 1-D sequence.")
        if np.any(indices < 0) or np.any(indices >= self.period):
            raise ValueError("circular changepoints must be in 0..period-1.")
        if np.any(np.diff(indices) <= 0):
            raise ValueError("circular changepoints must be sorted and duplicate-free.")
        object.__setattr__(self, "indices", indices)

    def rotated(self, offset: int) -> CircularChangePoints:
        """Return an equivalent changepoint set under a circular index rotation."""
        rotated = np.sort((self.indices + int(offset)) % self.period)
        return CircularChangePoints(period=self.period, indices=rotated)

    def segment_lengths(self) -> ArrayI:
        """Return segment lengths under periodic bin-end semantics."""
        if self.indices.size == 0:
            return np.array([self.period], dtype=int)
        lengths: list[int] = []
        previous = int(self.indices[-1])
        for cp in self.indices:
            distance = (int(cp) - previous) % self.period
            lengths.append(self.period if distance == 0 else distance)
            previous = int(cp)
        return np.asarray(lengths, dtype=int)

    def segments(self) -> tuple[CircularSegment, ...]:
        """Return circular segments as inclusive start/end bin pairs."""
        if self.indices.size == 0:
            return (CircularSegment(start=0, end=self.period - 1, length=self.period),)
        segments: list[CircularSegment] = []
        previous = int(self.indices[-1])
        for cp in self.indices:
            length = (int(cp) - previous) % self.period
            length = self.period if length == 0 else length
            segments.append(
                CircularSegment(
                    start=(previous + 1) % self.period,
                    end=int(cp),
                    length=length,
                )
            )
            previous = int(cp)
        return tuple(segments)
