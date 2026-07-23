# PELT Parameter Selection

PELT optimizes a cost plus changepoint penalty objective over half-open
right-exclusive segments.

## Cost Function Choice
- **NormalMeanKnownVar**: Gaussian profile deviance with known variance.
- **NormalMeanVarUnknown**: Gaussian profile deviance with optimized segment
  variance; length-one segments have infinite cost.
- **BetaBinomialCost**: negative log marginalized Bernoulli likelihood with a
  Beta prior.
- **Guideline**: match cost to data distribution; mismatch reduces accuracy.

## Penalty Parameter
- **AIC**: $2k$ in deviance units; matches the Gaussian costs.
- **BIC**: $k \log n$ in deviance units; matches the Gaussian costs.
- **Manual**: `pen=np.log(n)*variance` for domain-specific tuning.
- **Sensitivity**: Too small over-segments, too large misses changes.
- **Marginal costs**: tune penalties directly for `BetaBinomialCost`; the
  Gaussian AIC/BIC helpers are not on the same likelihood scale.

## Minimum Segment Length
- **Purpose**: Avoids spurious short segments.
- **Recommendation**: Set to expected shortest meaningful regime.

## Pruning Constant
- **Status**: retained for compatibility on the low-level `pelt(..., K=...)`
  API, but exact candidate retention is used for bundled costs.
- **Complexity**: assume $O(n^2)$ time unless a future benchmarked pruning path
  is enabled for the specific cost and minimum segment rule.

## Parameter Summary
| Parameter | Typical Range | Default | Notes |
|-----------|---------------|---------|-------|
| Cost function | n/a | NormalMeanVarUnknown | Choose according to data |
| Penalty | 0.5–10×$\log n$ | BIC | Tradeoff of over/under segmentation |
| Min segment length | 1–50 | 1 | Domain knowledge driven |
| Pruning constant | n/a | ignored | Retained for compatibility; exact candidate retention is used |

## Example: Penalty Sensitivity
```python
import numpy as np

from changepoint_lab import PELT
from changepoint_lab.algorithms.optimization.pelt import NormalMeanVarUnknown

x = np.concatenate([np.zeros(100), np.ones(100)])
for penalty in [5, 10, 20]:
    cost = NormalMeanVarUnknown()
    detector = PELT(cost_fn=cost, penalty=penalty)
    result = detector.fit_predict(x)
    print(penalty, result.indices)
```
Increasing penalty reduces detected changepoints; plotting F1 vs. penalty helps select a balanced value.

## Tuning
- Grid-search penalty with cross-validation.
- Compare Gaussian cost functions via AIC/BIC scores; tune marginal-likelihood
  costs on their own scale.

