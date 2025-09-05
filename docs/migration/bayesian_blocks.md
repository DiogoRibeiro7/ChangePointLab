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

Previous helpers `bayesian_blocks_counts` and `bayesian_blocks_events` do not yet
have direct counterparts in `changepoint_lab`. Support for these data types is
planned through extended BOCPD likelihoods. Until then, applications depending on
count or event-time segmentation should retain their existing implementation or
manually discretize data for use with other algorithms.

## Plotting

The `bb_plotting` utilities are no longer provided. Standard matplotlib code or
helpers in `changepoint_lab.common.plotting` can be used to visualize results.
