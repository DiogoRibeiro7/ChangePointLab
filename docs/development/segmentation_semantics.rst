Segmentation Semantics
======================

Date: 2026-07-23

ChangePointLab uses explicit boundary conventions for all public changepoint
results.

Linear data
-----------

The canonical linear convention is ``right_exclusive``.

- A changepoint ``k`` is the edge between observations ``k - 1`` and ``k``.
- The old segment ends at ``k``; the new segment starts at observation ``k``.
- Segment intervals are Python-style half-open slices ``[start, stop)``.
- Valid changepoints for length ``n`` are strictly increasing interior integers
  in ``1..n-1``.
- Endpoints ``0`` and ``n`` are segment edges, not changepoints.
- Duplicate and unsorted changepoints are invalid.
- Empty changepoint arrays represent one segment ``[0, n)``.
- Labels enumerate contiguous segments from left to right.

The conversion helpers in ``changepoint_lab.core.segmentation`` are the
canonical implementation:

- ``normalize_linear_changepoints``
- ``changepoints_to_edges``
- ``edges_to_changepoints``
- ``labels_from_changepoints``
- ``changepoints_from_labels``
- ``segment_slices``

PELT, E-Divisive, KernelCPD, HSMM, SD-HMM, and SD-HMM Mix VI wrappers return
linear ``right_exclusive`` boundaries when they emit ``ChangePointResult``
subclasses.

Online event indices
--------------------

BOCPD wrapper extraction uses ``time_index``. A returned index is the
observation time where the compatibility extraction rule is triggered. It is
not converted to a segment edge by default.

Circular data
-------------

Within-period changepoints are not linear segment edges. They use
``periodic_bin_end`` semantics through ``CircularChangePoints``.

- The period is a lattice ``0..period-1``.
- A changepoint ``k`` is the final bin of a circular segment.
- The next segment starts at ``(k + 1) % period``.
- Empty changepoints represent one full-period segment.
- Circular changepoints are sorted, duplicate-free integers in ``0..period-1``.
- Segment lengths are invariant under rotations of the period.

For example, with period ``24``, changepoints ``[7, 23]`` produce two segments:
bins ``0..7`` and bins ``8..23``. The changepoint at ``23`` marks the segment
that ends at midnight wrap-around; it is not a terminal linear boundary.
