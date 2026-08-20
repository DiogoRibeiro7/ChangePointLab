# ChangePointLab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/1046174252.svg)](https://zenodo.org/badge/latestdoi/1046174252)
[![CI](https://github.com/DiogoRibeiro7/ChangePointLab/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/ChangePointLab/actions/workflows/ci.yml)

Comprehensive Python toolkit for detecting structural changes in time series. The
package unifies multiple changepoint algorithms behind a common API, making it
straightforward to compare approaches and choose the right tool for a given data
set.

## Features

- Bayesian **online** detection for streaming data (BOCPD)
- Classical **offline** dynamic programming via **PELT**
- Non‑parametric **E‑Divisive** energy statistics
- Probabilistic **HSMM** models with explicit durations
- Kernel methods for non‑linear boundaries
- Within‑period detection for seasonal patterns
- Utilities for plotting, evaluation, and algorithm comparison

## Installation

Install from source:

```bash
git clone https://github.com/DiogoRibeiro7/ChangePointLab
cd ChangePointLab
poetry install
```

Requires Python 3.10 or later. Core installs require NumPy only. Plotting and
CSV time-binning helpers are optional:

```bash
poetry install --extras "plot data"
```

Development installs use `poetry install --with dev,docs --extras "plot data"`.

## Quick Start

<!-- docs-example: execute -->

```python
import numpy as np
from changepoint_lab import PELT
from changepoint_lab.algorithms.optimization.pelt import (
    NormalMeanVarUnknown,
    bic_penalty,
)

x = np.r_[np.zeros(50), np.ones(50)]
cost = NormalMeanVarUnknown()
cost.precompute(x)
detector = PELT(cost_fn=cost, penalty=bic_penalty(2, len(x)))
result = detector.fit_predict(x)
print(result.indices)
```

## Algorithms

| Algorithm | Description | Import Path |
|-----------|-------------|-------------|
| PELT | Offline exact segmentation | `from changepoint_lab import PELT` |
| BOCPD | Bayesian online detection | `from changepoint_lab import BOCPD` |
| EDivisive | Nonparametric energy statistics | `from changepoint_lab import EDivisive` |
| HSMM | Explicit‑duration state model | `from changepoint_lab import HSMM` |
| KernelCPD | Kernel‑based segmentation | `from changepoint_lab import KernelCPD` |
| WithinPeriodCPD | Seasonal Bayesian detector | `from changepoint_lab import WithinPeriodCPD` |

## Migration notes

Common import rewrites:

```python
from changepoint_lab import BOCPD, PELT
from changepoint_lab.algorithms.optimization.pelt import NormalMeanVarUnknown
```

Legacy module paths remain available but emit `DeprecationWarning` and will be
removed in version **0.3.0**.

## API stability

The top‑level classes re‑exported from `changepoint_lab` form the stable public
API. Deprecated aliases are scheduled for removal in version **0.3.0**.

## Documentation

See the `docs/` folder for tutorials and the API reference.

## Contributing

Contributions are welcome. Please read `CONTRIBUTING.md` before opening a pull
request, use the issue templates for reproducible reports, and report security
concerns through GitHub private vulnerability reporting rather than public
issues.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@article{ribeiro2025changepoint,
  title={ChangePointLab: A Unified Toolkit for Time Series Changepoint Detection},
  author={Ribeiro, Diogo F.},
  year={2026},
  publisher={Zenodo},
  note={See CITATION.cff and the Zenodo archive for release-specific metadata}
}
```

## Acknowledgments

Developed by Diogo Ribeiro and contributors at ESMAD - Instituto Politécnico do
Porto. We thank the open‑source community for foundational libraries such as
NumPy.

## References
- Adams, R., & MacKay, D. (2007). *Bayesian Online Changepoint Detection*.
- Matteson, D., & James, N. (2014). *A Nonparametric Approach for Multiple Change Point Analysis of Multivariate Data*.
- Killick, R., Fearnhead, P., & Eckley, I. (2012). *Optimal Detection of Changepoints With a Linear Computational Cost*.
- Yu, S.-Z. (2010). *Hidden Semi-Markov Models*.
