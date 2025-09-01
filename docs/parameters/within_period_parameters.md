# Within-Period Detection Parameter Selection

Within-period changepoint detection targets periodic data by resetting the hazard each cycle.

## Prior on Segment Counts
- **Interpretation**: Expected number of changepoints per period.
- **Guideline**: Use a Gamma prior with mean equal to historical average.

## Minimum Segment Length
- **Purpose**: Avoids detecting changes within short seasonal fluctuations.
- **Typical Range**: 1–10 time steps depending on sampling frequency.

## MCMC Tuning
- **Burn-in**: 10–20% of iterations discarded.
- **Thinning**: Keep every 5–10th sample to reduce autocorrelation.
- **Total Iterations**: 1k–10k depending on precision needs.

## Parallel Tempering
- **Configuration**: 3–5 chains with geometric temperature spacing.
- **Benefit**: Helps escape local modes in highly periodic data.

## Parameter Summary
| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Prior mean cp/period | 0.5–3 | Gamma prior on changepoint rate |
| Min segment length | 1–10 | Dependent on data resolution |
| Burn-in | 100–1000 | 10–20% of total iterations |
| Thinning | 5–10 | Keeps sample autocorrelation low |
| Iterations | 1000–10000 | More iterations for stable posteriors |
| Tempering chains | 3–5 | For multimodal posteriors |

## Example: MCMC Iterations vs. Convergence
```python
from within_period import WithinPeriodCPD
wp = WithinPeriodCPD(period=24)
wp.fit(x, iterations=2000, burn=200, thin=10)
```
Trace plots of log-posterior reveal when convergence is achieved; increasing iterations stabilizes changepoint estimates.

## Tuning
- Monitor effective sample size for key parameters.
- Adjust tempering and MCMC settings when chains fail to mix.

