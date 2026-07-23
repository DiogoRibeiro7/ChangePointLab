# Baseline Behaviour

Date: 2026-07-23
Baseline commit: `5bfd877`

This record freezes current behavior before scientific corrections. It does not
declare every current output scientifically correct. Each baseline is labelled
as:

- `scientific_oracle`: independently calculable expected behavior for a tiny
  case.
- `compatibility`: current behavior to preserve unless a planned migration says
  otherwise.
- `suspected_bug`: current behavior that is captured only so a future fix is
  deliberate and reviewable.

Fixture inputs live in `tests/fixtures/baseline/golden_inputs.json`. Recorded
outputs live in `tests/fixtures/baseline/current_outputs.json`.

## Public Wrappers

| Surface | Status | Baseline |
| --- | --- | --- |
| `PELT` | `scientific_oracle` | Known-variance Gaussian fixture has changepoint `[3]`; independent brute force confirms the optimum. |
| `BOCPD` | `compatibility` | Beta-Bernoulli run records `cp_prob`, MAP run lengths, predictive means, posterior shape, wrapper indices, and metadata keys. |
| `EDivisive` | `compatibility` | Tiny deterministic fixture with `R=9` and `seed=0` currently returns no accepted split. |
| `WithinPeriodCPD` | `compatibility` | Seeded small periodic binary fixture records three kept samples under the exact circular RJMCMC kernel and no longer emits singleton circular states. |
| `KernelCPD` | `compatibility` | Wrapper now executes and returns the right-exclusive interior changepoint `[2]` on the tiny kernel oracle. |
| `HSMM` | `scientific_oracle` | Core Viterbi oracle gives changepoint `[2]`; wrapper now extracts `[2]` from sparse duration-end indicators. |
| `SDHMM` | `compatibility` | Tiny compositional fixture records states `[1, 1, 0, 0]` and changepoint `[2]`. |
| `SDHMMMixVI` | `compatibility` | Tiny fixture now completes and returns states `[1, 1, 0, 0]` with changepoint `[2]`; scientific validation remains pending. |

## Low-Level Entry Points

| Surface | Status | Baseline |
| --- | --- | --- |
| `pelt` with `NormalMeanKnownVar` | `scientific_oracle` | Independent exhaustive segmentation over all changepoint subsets confirms `[3]`. |
| `pelt` with `NormalMeanVarUnknown` | `scientific_oracle` | Exact optimal partitioning now matches an independent brute-force oracle and returns changepoint `[3]` with score `-154.3505380291`. |
| `pelt_concave_penalty` | `scientific_oracle` | The local-linear penalty iteration delegates to the corrected exact PELT objective and returns changepoint `[3]` with score `-154.3505380291` on the fixture. |
| `kcp_penalized`, `kcp_fixed_m`, `kcp_select_bic` | `compatibility` | Exact KCP backtracking returns right-exclusive changepoint `[2]` with edges `[0, 2, 4]` on the tiny oracle. |
| `rff_kcp_penalized` | `compatibility` | RFF KCP backtracking drops the terminal endpoint and returns `[2]` on the deterministic tiny fixture. |
| within-period circular validity | `scientific_oracle` | Exhaustive enumeration for `N=6`, `l=2` is stored in fixtures and compared to implementation validity checks. |
| HSMM duration table and Viterbi | `scientific_oracle` | Independent truncated-Poisson duration probabilities and direct Viterbi state path confirm the core tiny case. |

## Determinism

The baseline test suite verifies selected deterministic outputs across two fresh
Python processes. This catches hidden dependence on process-local ordering or
global state for the frozen fixtures.

## Corrective Changes Recorded

- Within-period RJMCMC proposal accounting now has brute-force tiny-state
  detailed-balance tests, an explicit Poisson segment-count prior, and
  compatibility fixture updates for the corrected circular state space.
