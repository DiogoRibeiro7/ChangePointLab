# Migrating from `bayesian_blocks` to `changepoint_lab`

The standalone `bayesian_blocks` package has been removed in favor of the
unified `changepoint_lab` toolkit. This guide shows how to update existing code.

## Bernoulli/Binary data

```python
from bayesian_blocks import bayesian_blocks_bernoulli
result = bayesian_blocks_bernoulli(data, p0=0.05)
```

can be replaced with:

```python
from changepoint_lab.algorithms.bayesian.bocpd import BOCPD, BOCPDConfig, ConstantHazard

model = BOCPD(ConstantHazard(), cfg=BOCPDConfig())
result = model.fit(data).predict()
print(result.indices)
```

## Counts and event times

Scalar nonnegative count streams can use BOCPD with the Poisson-Gamma
likelihood:

```python
from changepoint_lab.algorithms.bayesian.bocpd import (
    BOCPD,
    BOCPDConfig,
    ConstantHazard,
    PoissonGamma,
)

model = BOCPD(
    ConstantHazard(mean_run_length=50),
    cfg=BOCPDConfig(max_run_length=200),
    likelihood=PoissonGamma(shape0=2.0, rate0=3.0),
)
result = model.run(counts)
```

Event-time segmentation does not have a direct Bayesian Blocks counterpart in
BOCPD. Use the sliced Poisson process detector for repeated event-time periods,
or retain the existing implementation for unbinned event-time Bayesian Blocks.

## Plotting

The `bb_plotting` utilities are no longer provided. Standard matplotlib code or
helpers in `changepoint_lab.common.plotting` can be used to visualize results.
