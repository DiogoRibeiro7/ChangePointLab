Result Invariants
=================

Stable result objects in ``changepoint_lab.core.datatypes`` validate their
public array fields when constructed or deserialized.

Changepoint arrays must be one-dimensional integer arrays with sorted, unique,
non-negative entries. Label-like arrays must be one-dimensional integer arrays
with non-negative entries. Cost, probability, posterior, and criterion arrays
must be one-dimensional and finite; probability arrays must stay within
``[0, 1]``.

Result subclasses also check dimensions for fields with shared semantics:

* ``SegmentationResult.costs_per_segment`` has one value per segment.
* ``OnlineProbabilityResult`` probability, run-length, and prediction arrays
  share a common non-empty length.
* ``LatentStateResult.segment_durations`` matches ``states`` when states are
  present.

All stable result arrays are defensively copied and made read-only. Mutating a
source array after construction does not change the result object. Consumers
that need a mutable array should copy the public array explicitly.

Deserialization validates ``result_type`` when present, plus known
``boundary_convention`` and ``objective_orientation`` values. Payloads produced
by ``to_dict()`` remain valid inputs for the matching ``from_dict()`` method.
