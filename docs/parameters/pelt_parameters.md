# PELT Parameter Selection

Pruned Exact Linear Time (PELT) optimizes a cost plus penalty objective.

## Cost Function Choice
- **NormalMeanKnownVar**: for Gaussian data with known variance.
- **NormalMeanVarUnknown**: handles unknown variance.
- **Binary Segmentation**: adapt cost to binomial or custom likelihoods.
- **Guideline**: match cost to data distribution; mismatch reduces accuracy.

## Penalty Parameter
- **AIC**: $2k$ favors recall.
- **BIC**: $k \log n$ favors precision.
- **Manual**: `pen=np.log(n)*variance` for domain-specific tuning.
- **Sensitivity**: Too small over-segments, too large misses changes.

## Minimum Segment Length
- **Purpose**: Avoids spurious short segments.
- **Recommendation**: Set to expected shortest meaningful regime.

## Pruning Constant
- **Controls** early termination of candidate changepoints.
- **Typical Range**: 0–10; larger values increase runtime but may improve accuracy.

## Parameter Summary
| Parameter | Typical Range | Default | Notes |
|-----------|---------------|---------|-------|
| Cost function | n/a | NormalMeanVarUnknown | Choose according to data |
| Penalty | 0.5–10×$\log n$ | BIC | Tradeoff of over/under segmentation |
| Min segment length | 1–50 | 1 | Domain knowledge driven |
| Pruning constant | 0–10 | 5 | Larger = slower, more accurate |

## Example: Penalty Sensitivity
```python
from changepoint_lab.algorithms.optimization.pelt import pelt
import numpy as np
x = np.concatenate([np.zeros(100), np.ones(100)])
for penalty in [5, 10, 20]:
    cps = pelt(x, penalty=penalty)
    print(penalty, cps)
```
Increasing penalty reduces detected changepoints; plotting F1 vs. penalty helps select a balanced value.

## Tuning
- Grid-search penalty with cross-validation.
- Compare different cost functions via AIC/BIC scores.

