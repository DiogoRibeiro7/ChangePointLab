# SD-HMM Parameter Selection

Switching Dirichlet HMMs model compositional observations with simplex constraints.

## Component Count
- **Meaning**: Number of mixture components per state.
- **Selection**: Start with 1–3 components; increase if residuals show multimodality.

## Dirichlet Parameters
- **Role**: Concentration parameters controlling component sparsity.
- **Guideline**: Values <1 encourage corner solutions; >1 promote uniformity.
- **Sensitivity**: Extreme values can cause numerical instability near simplex boundaries.

## Regularization
- **Techniques**: Add small pseudo-counts or L2 penalties on logit-transformed weights.
- **Purpose**: Prevents zero probabilities and improves convergence.

## Initialization
- **Approaches**: Use k-means on centered log-ratio transformed data or random simplex points.
- **Multiple Restarts**: Recommended due to multimodal likelihood.

## Parameter Summary
| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Components/state | 1–5 | Larger increases flexibility but costs runtime |
| Dirichlet $\alpha$ | 0.1–10 | <1 sparse, >1 uniform |
| Regularization | $10^{-6}$–$10^{-2}$ | Added to avoid zeros |

## Example: Dirichlet Concentration Effect
```python
from changepoint_lab import SDHMM
model = SDHMM(n_states=2, components=2, alpha=[0.2,0.2])
model.fit(X)
```
Increasing `alpha` produces smoother state distributions; plotting state probabilities shows shrinkage toward the center of the simplex.

## Tuning
- Evaluate held-out log-likelihood and compositional reconstruction error.
- Use cross-validation to choose component count and regularization.

