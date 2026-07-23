# 0013: PELT cost and penalty oracle validation

Date: 2026-07-23

## Decision

Validate the bundled PELT costs against an independently written exhaustive
optimal-partitioning oracle and prioritize exact objective correctness over
pruning speed claims.

The low-level `pelt(..., K=...)` argument remains accepted for compatibility,
but the implementation now retains exact candidates for bundled costs. Penalty
helpers `aic_penalty` and `bic_penalty` are documented as deviance-scale
helpers matching the Gaussian costs; `BetaBinomialCost` uses negative log
marginal likelihood and should be tuned on its own scale.

## Consequences

- Gaussian known-variance, Gaussian unknown-variance, and beta-binomial PELT
  outputs are checked against exhaustive small-sequence oracles.
- The previous unknown-variance baseline with no changepoint was corrected to
  changepoint `[3]` on the tiny fixture.
- Complexity documentation no longer claims unconditional linear-time behavior.
- `pelt_concave_penalty` remains an approximation because it solves repeated
  linearized objectives rather than the original concave-penalty problem
  exactly.
