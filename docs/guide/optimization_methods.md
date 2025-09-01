# Optimization-Based Methods: PELT

Optimization approaches frame changepoint detection as minimizing a cost over all possible segmentations.

## Cost Functions
A segmentation $\{\tau_j\}$ has total cost
$$C = \sum_{j} \mathcal{C}(x_{(\tau_{j-1}+1):\tau_j}) + \beta \cdot (m-1),$$
where $\mathcal{C}$ is a segment cost and $\beta$ a penalty controlling the number of changepoints.
Common costs include:
- **NormalMeanKnownVar**: squared error with known variance.
- **NormalMeanVarUnknown**: uses sample variance per segment.

## Penalty Tuning
- Larger $\beta$ yields fewer changepoints; smaller values increase sensitivity.
- Information criteria (AIC, BIC) provide data-driven defaults.

## Pruning and Efficiency
PELT uses the **pruned exact linear time** principle: segments violating a pruning condition are discarded, yielding $O(n)$ complexity under mild assumptions.

## Interpreting Optimal Segmentations
The optimal changepoint set balances fit and complexity. Visual inspection and domain knowledge remain essential to validate results.

See [PELT API](../api/pelt.rst) for implementation details and cost/penalty options.
