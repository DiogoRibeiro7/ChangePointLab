# Optimization-Based Methods: PELT

Optimization approaches frame changepoint detection as minimizing a cost over all possible segmentations.

## Cost Functions
A segmentation $\{\tau_j\}$ has total cost
$$C = \sum_{j} \mathcal{C}(x_{(\tau_{j-1}+1):\tau_j}) + \beta \cdot (m-1),$$
where $\mathcal{C}$ is a segment cost and $\beta$ a penalty controlling the number of changepoints.
Common costs include:
- **NormalMeanKnownVar**: Gaussian profile deviance with known variance.
- **NormalMeanVarUnknown**: Gaussian profile deviance with optimized segment variance; length-one segments have infinite cost.
- **BetaBinomialCost**: negative log marginalized Bernoulli likelihood with a Beta prior.

## Penalty Tuning
- Larger $\beta$ yields fewer changepoints; smaller values increase sensitivity.
- The `aic_penalty` and `bic_penalty` helpers are in deviance units, matching
  the Gaussian costs. Use direct tuned penalties for marginalized costs such as
  `BetaBinomialCost`.

## Pruning and Efficiency
The implementation returns the exact global optimum of the penalized objective.
Candidate pruning is currently retained only as API terminology; bundled costs
and `min_seg_len` constraints use exact candidate retention to avoid invalid
pruning. Treat runtime as $O(n^2)$ unless benchmark evidence for a specific
cost and configuration shows effective pruning.

`pelt_concave_penalty` is an approximation: it repeatedly linearizes a concave
penalty and solves the resulting linear-penalty objective. It is not guaranteed
to find the global optimum of the original concave objective.

## Interpreting Optimal Segmentations
The optimal changepoint set balances fit and complexity. Visual inspection and domain knowledge remain essential to validate results.

See [PELT API](../api/pelt.rst) for implementation details and cost/penalty options.
