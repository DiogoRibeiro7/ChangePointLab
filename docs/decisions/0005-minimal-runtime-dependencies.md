# 0005: Minimal runtime dependencies and optional extras

Date: 2026-07-23

## Decision

ChangePointLab core runtime depends only on NumPy. Matplotlib is provided by the
`plot` extra, and pandas is provided by the `data` extra.

## Context

The package metadata previously required Matplotlib and pandas for every
install, even though the core algorithms are NumPy-based. Public imports also
pulled plotting modules transitively, which made Matplotlib mandatory.

## Consequences

- `import changepoint_lab` works without plotting or data-frame packages.
- Plotting and CSV time-binning code use lazy optional imports with clear
  installation messages.
- CI tests both core-only compatibility and optional-feature coverage.
- CLI commands remain installed by default; commands that produce plots or read
  pandas-backed CSV time bins require the relevant extras at execution time.
