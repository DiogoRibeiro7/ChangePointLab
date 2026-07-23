# Decision 0001: Audit Before Behavioral Modernization

Date: 2026-07-23

## Context

ChangePointLab exposes multiple changepoint algorithms with different
mathematical assumptions, result contracts, index semantics, and randomness
behavior. The repository also contains compatibility shims, packaged examples,
CLI wrappers, paper material, and broad documentation claims.

Several public paths appear broken or scientifically under-validated. Changing
algorithm behavior before freezing the current surface would make it hard to
distinguish compatibility-preserving refactors from scientific corrections.

## Decision

Perform repository inventory, risk registration, and dependency-aware roadmap
work before changing production algorithm behavior.

The first modernization pass may add audit documentation and validation records,
but must not change algorithm defaults, objective functions, index conventions,
randomness, tolerances, or returned values.

## Consequences

- Broken public paths are documented as risks instead of fixed immediately.
- Scientific fixes must wait for characterization tests and independent oracles.
- The roadmap orders traceability and baseline work ahead of API redesign,
  optimization, documentation polish, and release expansion.
- GitHub/Zenodo releases remain allowed only for coherent, verified repository
  states; PyPI publishing and JOSS submission remain out of scope.
