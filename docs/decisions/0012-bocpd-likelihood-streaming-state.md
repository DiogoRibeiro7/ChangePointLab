# 0012: BOCPD likelihood support and streaming state

Date: 2026-07-23

## Decision

Support two BOCPD likelihood families publicly:

- `BetaBernoulli` for binary streams.
- `PoissonGamma` for scalar nonnegative integer count streams.

`BOCPD` accepts an explicit likelihood instance, clones that model on `reset()`,
and exposes `update`, `update_many`, `state_dict`, and `load_state_dict` for
online processing and checkpoint/resume.

## Context

The previous implementation exposed a pluggable likelihood abstraction but always
reset to Beta-Bernoulli. `PoissonGamma` and `GaussianNIW` were placeholders, while
some documentation implied broader Gaussian or Student-t BOCPD support.

## Consequences

- Reset no longer silently replaces an injected likelihood.
- Batch processing and repeated online updates have a shared contract.
- Checkpoint/resume is supported when restoring into a model with the same
  hazard, configuration, and likelihood family.
- Missing observations (`None` or all-NaN numeric values) advance the run-length
  transition without updating likelihood sufficient statistics.
- Gaussian and Student-t BOCPD claims remain future work until implemented with
  independent oracle tests.
