# Within-Period Detection Parameter Selection

Within-period changepoint detection targets binary observations repeated over a fixed period. Boundaries are reported as periodic bin-end indices modulo `N`.

## Prior on Segment Counts
- **Interpretation**: `pois_lambda` controls the truncated Poisson prior on the number of circular segments `m`, supported on `1..floor(N / l)`.
- **Guideline**: Use values near 1 for a sparse segmentation prior. Larger values increase posterior support for more segments when the data do not dominate the prior.

## Minimum Segment Length
- **Purpose**: Avoids detecting changes within short seasonal fluctuations.
- **Typical Range**: 1–10 time steps depending on sampling frequency.
- **State space**: The one-segment model is represented by `tau=()`. Non-empty states store one circular boundary per segment, so singleton `tau` states are invalid.

## MCMC Tuning
- **Burn-in**: 10–20% of iterations discarded.
- **Thinning**: Keep every 5–10th sample to reduce autocorrelation.
- **Total Iterations**: 1k–10k depending on precision needs.
- **Proposal weights**: `move_prob`, `birth_prob`, and `death_prob` must be positive and sum to 1. At states where a proposal family is impossible, the sampler renormalizes the available families.

## Parallel Tempering
- **Configuration**: The helper currently runs a two-chain cold/hot sampler.
- **Benefit**: Helps escape local modes in highly periodic data.

## Parameter Summary
| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| `pois_lambda` | 0.5–3 | Truncated Poisson prior on segment count |
| Min segment length | 1–10 | Dependent on data resolution |
| Burn-in | 100–1000 | 10–20% of total iterations |
| Thinning | 5–10 | Keeps sample autocorrelation low |
| Iterations | 1000–10000 | More iterations for stable posteriors |
| Hot temperature | 2–5 | For multimodal posteriors |

## Example: MCMC Iterations vs. Convergence
```python
from changepoint_lab.algorithms.bayesian.within_period import ModelPrior, RJConfig, WithinPeriodCPD

prior = ModelPrior(N=24, l=2, gamma=1.0, pois_lambda=1.0)
cfg = RJConfig(iters=2000, burn=200, thin=10, seed=7)
wp = WithinPeriodCPD(prior, cfg=cfg).fit(x)
```
Trace plots of log-posterior reveal when convergence is achieved; increasing iterations stabilizes changepoint estimates.

## Tuning
- Monitor effective sample size for key parameters.
- Inspect `result.acceptance_rate` and `result.move_counts` for proposal balance.
- Adjust tempering and MCMC settings when chains fail to mix.

