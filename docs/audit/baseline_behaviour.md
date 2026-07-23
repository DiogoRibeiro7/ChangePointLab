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
| `WithinPeriodCPD` | `compatibility` | Seeded small periodic binary fixture records three kept samples and mode `()`. |
| `KernelCPD` | `suspected_bug` | Wrapper raises `AttributeError: 'tuple' object has no attribute 'K'` because `gram_rbf` returns `(K, gamma)` while the wrapper passes it directly to `kcp_penalized`. |
| `HSMM` | `suspected_bug` | Core Viterbi oracle gives changepoint `[2]`; wrapper currently reports `[0, 2, 2]` from sparse duration-end indicators. |
| `SDHMM` | `compatibility` | Tiny compositional fixture records states `[1, 1, 0, 0]` and changepoint `[2]`. |
| `SDHMMMixVI` | `suspected_bug` | Tiny fixture raises `TypeError: 'tuple' object does not support item assignment`. |

## Low-Level Entry Points

| Surface | Status | Baseline |
| --- | --- | --- |
| `pelt` with `NormalMeanKnownVar` | `scientific_oracle` | Independent exhaustive segmentation over all changepoint subsets confirms `[3]`. |
| `pelt` with `NormalMeanVarUnknown` | `compatibility` | Current fixture returns no changepoint and score `25.3450285652`. |
| `pelt_concave_penalty` | `compatibility` | Current fixture returns no changepoint and score `25.3450285652`. |
| `kcp_penalized`, `kcp_fixed_m`, `kcp_select_bic` | `suspected_bug` | Independent kernel oracle finds `[2]`, but current package outputs terminal changepoint `[4]` with edges `[0, 4, 4]`. |
| `rff_kcp_penalized` | `suspected_bug` | Current RFF path also emits terminal changepoint `[4]`. |
| within-period circular validity | `scientific_oracle` | Exhaustive enumeration for `N=6`, `l=2` is stored in fixtures and compared to implementation validity checks. |
| HSMM duration table and Viterbi | `scientific_oracle` | Independent truncated-Poisson duration probabilities and direct Viterbi state path confirm the core tiny case. |

## Determinism

The baseline test suite verifies selected deterministic outputs across two fresh
Python processes. This catches hidden dependence on process-local ordering or
global state for the frozen fixtures.

## Expected Future Changes

The following current outputs may change under a planned correctness fix:

- `KernelCPD` wrapper should stop raising and should not pass a `(K, gamma)`
  tuple where a prefix object is expected.
- KCP/RFF low-level backtracking should not emit terminal changepoints equal to
  `n`.
- HSMM wrapper should convert sparse per-end duration indicators into segment
  durations before computing changepoints.
- `SDHMMMixVI` should update immutable parameter tuples safely.
- Within-period RJMCMC should avoid global RNG and should handle small valid
  circular cases without `math domain error`.

Any future correction to these paths should update the `suspected_bug` fixture
entries and include migration notes where public outputs change.
