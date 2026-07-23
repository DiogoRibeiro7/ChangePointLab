# Decision 0003: Freeze Baselines Before Corrections

Date: 2026-07-23

## Context

The audit and method registry identified public wrappers and low-level
algorithms that either lack independent scientific oracles or show likely
correctness defects. Fixing them without a frozen baseline would make it harder
to tell whether a change is deliberate correction, accidental regression, or
compatibility break.

## Decision

Store small deterministic fixtures and expected outputs before making numerical
or statistical corrections. Baselines must be explicitly labelled as
`scientific_oracle`, `compatibility`, or `suspected_bug`.

Scientific oracles must be computed independently from the implementation under
test. Suspected-bug baselines are allowed only to document current behavior and
must not be treated as future desired behavior.

## Consequences

- Future fixes can update tests from `suspected_bug` to corrected oracle
  behavior in one deliberate change.
- Compatibility outputs have recorded shapes, metadata keys, warnings, and
  exceptions.
- CI now guards the fixture files against local paths, wall-clock drift, and
  platform-specific path separators.
