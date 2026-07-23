# Decision 0002: Evidence-First Scientific Claims

Date: 2026-07-23

## Context

ChangePointLab contains multiple literature-derived algorithms and historical
documentation that made broad claims about method coverage, benchmark
performance, publication status, and release channels.

The current release scope is GitHub plus Zenodo. PyPI publishing and JOSS
submission are out of scope. Some implemented methods are characterized by tests
but still lack independent scientific oracles.

## Decision

Maintain a machine-readable method registry and claim audit before making or
restoring scientific claims. Active documentation must distinguish implemented
behavior from adaptations, extensions, aspirational plans, and missing methods.

Unsupported benchmark, publication, and release-channel claims are removed from
active documentation only after being recorded in `docs/science/claim_audit.md`.

## Consequences

- Methods can be exposed while still marked below `verified`.
- Numeric performance or accuracy claims require generated benchmark artifacts.
- The sliced Poisson process is recorded as research scope, but not implemented.
- Zenodo remains the only external release/archive target in the current scope.
