# Legacy cpd package

The former `cpd` package has been split into dedicated modules:

- `WithinPeriodCPD` – within-period change-point detection
- `algorithms.kernel` – kernel change-point detection and RFF helpers
- `hsmm` – state-space emissions
- `toolkit` – shared CLI and API glue
- `common` – utilities, I/O, plotting, and diagnostics

This file exists for historical context only.

## Advanced features

### Random Fourier feature variants

The `changepoint_lab.algorithms.kernel.rff_variants` module provides orthogonal, quasi–Monte Carlo, and
compact-support RFF mappings.  A minimal usage example:

```python
from changepoint_lab.algorithms.kernel.rff_variants import (
    orthogonal_rff_map,
    OrthogonalRFFConfig,
)

Z = orthogonal_rff_map(X, OrthogonalRFFConfig(n_features=512)).Z
```

Increasing `n_features` improves accuracy but scales linearly in memory and
runtime.

### Parallel tempering sampler

Within-period change-point models can leverage parallel tempering via
`changepoint_lab.algorithms.bayesian.within_period.samplers.tempering`.  Each additional temperature requires an
extra Markov chain, so the method scales roughly linearly with the number of
temperatures.

```python
from changepoint_lab.algorithms.bayesian.within_period.samplers.tempering import tempering_sampler

for state in tempering_sampler(model, n_temps=4):
    ...  # consume samples
```

These advanced options can dramatically increase computational cost; consider
starting with small configurations before scaling up.
