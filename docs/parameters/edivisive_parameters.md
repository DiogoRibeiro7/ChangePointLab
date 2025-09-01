# E-Divisive Parameter Selection

E-Divisive detects changepoints via energy statistics without distributional assumptions.

## Alpha for Distance Metrics
- **Role**: Exponent on Euclidean distance in energy statistic.
- **Typical Values**: 1 (absolute) or 2 (squared).
- **Sensitivity**: Higher values emphasize large deviations; lower values robust to outliers.

## Significance Threshold
- **Purpose**: Critical value for permutation test.
- **Recommendation**: 0.05 for general use; lower for stricter detection.

## Permutation Count
- **Effect**: More permutations yield stable $p$-values but increase runtime.
- **Guideline**: 100–1000 permutations; use 500 as default.

## Minimum Segment Length
- **Purpose**: Enforces minimum number of observations per segment.
- **Typical Range**: 5–30 depending on data frequency.

## Parameter Summary
| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| `alpha` | 1–2 | 1 | Exponent for distance metric |
| `significance` | 0.01–0.1 | 0.05 | Permutation $p$-value cutoff |
| `permutations` | 100–1000 | 500 | Number of permutations |
| `min_size` | 5–30 | 10 | Minimum segment length |

## Example: Permutation Count vs. Runtime
```python
from edivisive import edivisive
import numpy as np, time
x = np.concatenate([np.random.normal(0,1,100), np.random.normal(1,1,100)])
for perms in [100, 500, 1000]:
    t0 = time.time(); edivisive(x, permutations=perms); dt = time.time()-t0
    print(perms, dt)
```
Runtime grows roughly linearly with permutation count, while detection power stabilizes beyond ~500 permutations.

## Tuning
- Perform permutation count vs. accuracy tradeoff using validation sets.
- Adjust `alpha` via cross-validation when heavy-tailed noise is present.

