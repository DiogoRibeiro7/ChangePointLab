# HMM/HSMM Parameter Selection

Hidden Markov Models (HMM) and Hidden Semi-Markov Models (HSMM) require careful parameter choices for reliable changepoint detection.

## State Count
- **Strategies**:
  - Use information criteria (AIC/BIC) on fitted models.
  - Cross-validate log-likelihood or segmentation accuracy.
- **Sensitivity**: Too few states merge regimes; too many overfit noise.

## Duration Distributions (HSMM)
- **Options**: Poisson, geometric, or nonparametric.
- **Guideline**: Match empirical dwell-time; over-dispersed data benefit from negative binomial or custom durations.

## Emission Model Tuning
- Select Gaussian, Bernoulli, or other emissions consistent with data type.
- Regularize covariance estimates for high-dimensional data.

## Initialization
- K-means or segment-based initialization stabilizes EM.
- Multiple random restarts reduce local optima risk.

## Convergence Criteria
- **Tolerance** on log-likelihood change (e.g., $10^{-4}$).
- **Max Iterations**: 100–1000; monitor overfitting with validation likelihood.

## Parameter Summary
| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| `n_states` | 2–10 | Start small; expand as needed |
| Duration params | Domain dependent | Choose distribution matching dwell-time |
| Emission params | Data-driven | Regularize for stability |
| `tol` | $10^{-6}$–$10^{-3}$ | EM convergence threshold |
| `n_iter` | 100–1000 | Max EM iterations |

## Example: State Count vs. BIC
```python
from hsmm import HSMM
scores = []
for k in range(2,6):
    model = HSMM(n_states=k).fit(X)
    scores.append(model.bic_)
plt.plot(range(2,6), scores)
```
Choose the state count at the BIC elbow.

## Tuning
- Evaluate out-of-sample likelihood.
- Compare HMM vs. HSMM if dwell-time distribution is uncertain.

