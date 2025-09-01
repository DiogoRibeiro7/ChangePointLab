# BOCPD Parameter Selection

Bayesian Online Changepoint Detection (BOCPD) requires several hyperparameters that control sensitivity and computational cost.

## Mean Run Length
- **Purpose**: Sets expected segment duration for `ConstantHazard`.
- **Selection**: Choose near the median segment length; shorter values detect rapid changes but raise false positives.
- **Sensitivity**: F1 peaks around the true mean; performance drops when off by >50%.

## Alpha/Beta Priors
- **Role**: Beta-Bernoulli or Beta-Binomial priors encode prior event rates.
- **Tuning**:
  - Small $(\alpha,\beta)\approx(1,1)$ adapts quickly but is noisy.
  - Large values damp updates; set pseudo-counts equal to expected successes/failures in one segment.
- **Example**:
```python
prior = dict(alpha=2, beta=8)  # baseline success rate 0.2
```

## Hazard Function Selection
- **ConstantHazard**: For memoryless changepoints.
- **ScheduledHazard**: Use when changes align with known times.
- **BoostedBoundaryHazard**: Amplifies detection near boundaries or periodic indices.

## Maximum Run Length
- **Purpose**: Caps posterior table size.
- **Guideline**: At least twice the longest expected segment; too small biases towards frequent changepoints.

## Pruning Parameter
- **`cp_scale`** rescales changepoint probabilities before pruning.
- **Effect**: Larger values increase sensitivity by retaining more candidates.

## Parameter Summary
| Parameter | Typical Range | Default | Notes |
|-----------|---------------|---------|-------|
| `mean_run_length` | 10–1000 | 200 | Memoryless changepoint rate |
| `alpha`, `beta` | 0.1–100 | 1 | Prior counts for successes/failures |
| `max_run_length` | 50–5000 | 1000 | Posterior truncation |
| `cp_scale` | 0.5–5 | 1.0 | Multiplicative scaling before pruning |

## Example: Mean Run Length Sensitivity
```python
import numpy as np, matplotlib.pyplot as plt
from bocpd import BOCPD, ConstantHazard

x = np.concatenate([np.zeros(50), np.ones(50)])
for mrl in [25, 50, 100]:
    model = BOCPD(hazard=ConstantHazard(mrl))
    cps = model.fit_predict(x)
    plt.plot(model.cp_prob, label=f"mrl={mrl}")
plt.legend(); plt.show()
```
Running the above shows that underestimating `mean_run_length` yields spurious spikes, while overestimation delays detection.

## Tuning
- Use cross-validation on historical data.
- Evaluate precision/recall vs. runtime to balance sensitivity and cost.

