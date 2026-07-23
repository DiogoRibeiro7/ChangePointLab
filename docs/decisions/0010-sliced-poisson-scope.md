# 0010 - Sliced Poisson process scope

Date: 2026-07-23

## Status

Accepted

## Context

Martinez-Hernandez and Killick (2024) model each day of event times as one
inhomogeneous Poisson process observation and segment the sequence of daily
processes. The paper uses B-spline log-intensity representations and a
penalized cost approach. It also discusses possible multivariate extensions,
but leaves those extensions as future work.

## Decision

Implement the faithful baseline as an unmarked IHPP sliced Poisson detector:

- event times are repeated periods on `[0, period)`;
- optional exposure intervals represent observed windows;
- segment cost is optimized minus twice IHPP log-likelihood;
- the shared PELT interface is used for the across-period additive objective;
  its `K` argument is currently compatibility-only because exact candidate
  retention is used for bundled costs;
- marked sensors are available only through an explicit independent-by-mark
  helper.

The shared-baseline marked extension is not implemented and raises
`NotImplementedError`.

## Consequences

The repository now has a dedicated public API for the sliced Poisson method and
tests for analytical integrals, simulated recovery, exposure handling, and
marked extension boundaries. It does not claim parity with the paper's private
Howz data or supplementary code.
