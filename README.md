# Change-Point Detection Toolkit

Comprehensive toolkit for change-point detection in time series.

## Installation

```bash
pip install changepoint-toolkit
```

## Quick start

```python
import numpy as np
from changepoint_toolkit import BOCPD, BOCPDConfig, ConstantHazard

# generate synthetic binary data
rng = np.random.default_rng(0)
data = rng.random(100) > 0.5

model = BOCPD(ConstantHazard(mean_run_length=100), BOCPDConfig())
for x in data:
    result = model.update(int(x))
print(result["cp_prob"])
```

## Algorithms

- **BOCPD** – Bayesian online change-point detection for streaming binary data.
- **WithinPeriodCPD** – Periodic change-point detection via reversible-jump MCMC.
- **kcp_penalized** – Kernel change-point detection with penalized dynamic programming.
- **kcp_select_bic** – Kernel change-point model selection using BIC.
- **edivisive** – Nonparametric divisive detection using energy statistics.
- **pelt** – Pruned Exact Linear Time algorithm for multiple change points.
- **HSMM** – Hidden semi-Markov model for change-point inference.
- **SDHMM** – Switching-duration hidden Markov model with explicit durations.

## Documentation

- [Documentation index](docs/)
- [BOCPD guide](bocpd/README.md)
- [Legacy CPD notes](docs/cpd.md)
