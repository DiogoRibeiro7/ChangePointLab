# 0011: BOCPD posterior calibration and alert extraction

Date: 2026-07-23

## Decision

Change BOCPD defaults to the unscaled Adams-MacKay run-length recursion and move
changepoint event extraction into `BOCPDAlertConfig` post-processing.

## Context

The previous configuration included `cp_scale`, a multiplier applied directly to
the changepoint transition mass before normalization. That made `cp_prob`
incompatible with the documented interpretation `P(r_t = 0 | x_1:t)`. The
estimator wrapper also used an implicit `cp_prob > 0.5` rule for returned
indices.

## Consequences

- `cp_prob` has a probabilistic interpretation when `cp_scale == 1.0`.
- Wrapper and CLI alert indices are configurable thresholding policies, not part
  of the Bayesian recursion.
- `cp_scale != 1.0` remains available temporarily for compatibility comparisons,
  emits `DeprecationWarning`, and marks diagnostics as not calibrated.
- Pruning and top-k modes report removed posterior mass through
  `approximation_error`.
