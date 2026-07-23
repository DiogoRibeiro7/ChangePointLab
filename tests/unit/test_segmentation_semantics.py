from __future__ import annotations

import itertools

import numpy as np
import pytest

from changepoint_lab.core.segmentation import (
    CircularChangePoints,
    changepoints_from_labels,
    changepoints_to_edges,
    edges_to_changepoints,
    labels_from_changepoints,
    normalize_linear_changepoints,
    segment_slices,
)


def _all_partitions(n: int):
    for mask in range(1 << max(0, n - 1)):
        yield [idx + 1 for idx in range(n - 1) if mask & (1 << idx)]


def test_linear_conversions_round_trip_all_small_partitions() -> None:
    for n in range(1, 8):
        for cps in _all_partitions(n):
            edges = changepoints_to_edges(cps, n=n)
            labels = labels_from_changepoints(n, cps)
            slices = segment_slices(n, cps)

            assert edges_to_changepoints(edges, n=n).tolist() == cps
            assert changepoints_from_labels(labels).tolist() == cps
            assert [sl.stop - sl.start for sl in slices] == np.diff(edges).tolist()


def test_linear_endpoint_duplicate_and_sort_rules_are_explicit() -> None:
    assert normalize_linear_changepoints([], n=1).tolist() == []
    assert labels_from_changepoints(4, [1, 2, 3]).tolist() == [0, 1, 2, 3]

    with pytest.raises(ValueError, match="interior"):
        normalize_linear_changepoints([0], n=4)
    with pytest.raises(ValueError, match="interior"):
        normalize_linear_changepoints([4], n=4)
    with pytest.raises(ValueError, match="strictly increasing"):
        normalize_linear_changepoints([2, 2], n=4)
    with pytest.raises(ValueError, match="strictly increasing"):
        normalize_linear_changepoints([3, 1], n=4)
    with pytest.raises(ValueError, match="min_segment_length"):
        normalize_linear_changepoints([1], n=4, min_segment_length=2)


def test_circular_periodic_bin_end_segments_rotate_and_wrap() -> None:
    circular = CircularChangePoints(period=6, indices=np.array([1, 4]))
    assert circular.boundary_convention == "periodic_bin_end"
    assert circular.segment_lengths().tolist() == [3, 3]
    assert [(s.start, s.end, s.length) for s in circular.segments()] == [
        (5, 1, 3),
        (2, 4, 3),
    ]

    for offset in range(6):
        assert circular.rotated(offset).segment_lengths().tolist() == [3, 3]

    midnight_wrap = CircularChangePoints(period=24, indices=np.array([7, 23]))
    assert [(s.start, s.end, s.length) for s in midnight_wrap.segments()] == [
        (0, 7, 8),
        (8, 23, 16),
    ]

    with pytest.raises(ValueError, match="0..period-1"):
        CircularChangePoints(period=24, indices=np.array([24]))
    with pytest.raises(ValueError, match="sorted and duplicate-free"):
        CircularChangePoints(period=24, indices=np.array([7, 7]))


def test_circular_lengths_are_rotation_invariant_for_small_partitions() -> None:
    for period in range(2, 8):
        for count in range(period):
            for cps in itertools.combinations(range(period), count):
                circular = CircularChangePoints(period=period, indices=np.array(cps))
                lengths = sorted(circular.segment_lengths().tolist())
                for offset in range(period):
                    assert sorted(circular.rotated(offset).segment_lengths().tolist()) == lengths
