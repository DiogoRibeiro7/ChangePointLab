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
- **Guideline**: use `R=499` for routine CLI/low-level runs when cost permits; smaller values are useful for tests and exploratory work.

## Resampling Mode
- **IID permutation**: `resample="iid"` is the direct exchangeability assumption.
- **Block permutation**: `resample="block-permutation"` shuffles non-overlapping contiguous blocks and preserves within-block dependence.
- **Circular block bootstrap**: `resample="circular-block-bootstrap"` samples moving circular blocks with replacement and is an extension for dependent sequences.

## Minimum Segment Length
- **Purpose**: Enforces minimum number of observations per segment.
- **Typical Range**: 5–30 depending on data frequency.

## Parameter Summary
| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| `alpha` | 1–2 | 1 | Exponent for distance metric |
| `significance` | 0.01–0.1 | 0.05 | Permutation $p$-value cutoff |
| `R` | 100–1000 | 499 low-level / 199 wrapper | Number of resamples |
| `min_size` | 5–30 | 10 | Minimum segment length |
| `resample` | iid, block-permutation, circular-block-bootstrap | iid | Null resampling scheme |
| `block_size` | >=2 | auto | Block length for dependent-data resampling |

## Example: Permutation Count vs. Runtime
```python
from changepoint_lab import edivisive
import numpy as np, time
x = np.concatenate([np.random.normal(0,1,100), np.random.normal(1,1,100)])
for perms in [100, 500, 1000]:
    t0 = time.time(); edivisive(x, R=perms); dt = time.time()-t0
    print(perms, dt)
```
Runtime grows roughly linearly with permutation count, while detection power stabilizes beyond ~500 permutations.

## Tuning
- Perform permutation count vs. accuracy tradeoff using validation sets.
- Adjust `alpha` via cross-validation when heavy-tailed noise is present.

