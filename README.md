# Change-Point Detection Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/changepoint-toolkit.svg)](https://pypi.org/project/changepoint-toolkit/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.0000000.svg)](https://doi.org/10.5281/zenodo.0000000)
[![JOSS](https://joss.theoj.org/papers/10.21105/joss.00000/status.svg)](https://doi.org/10.21105/joss.00000)
[![CI](https://github.com/DiogoRibeiro7/articles_future/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/articles_future/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/DiogoRibeiro7/articles_future/branch/main/graph/badge.svg)](https://codecov.io/gh/DiogoRibeiro7/articles_future)

Comprehensive Python toolkit for detecting structural changes in time series. The
package unifies multiple change-point algorithms behind a common API, making it
straightforward to compare approaches and choose the right tool for a given
data set.

## Features

- Bayesian **online** detection for streaming data (BOCPD)
- Classical **offline** dynamic programming via **PELT**
- Non-parametric **E-Divisive** energy statistics
- Probabilistic **HMM/HSMM** models with explicit durations
- Scaled-Dirichlet **SD-HMM** for compositional data
- **Within-period** detection for repeating seasonal patterns
- Utilities for plotting, evaluation, and algorithm comparison

## Method comparison

| Method | Setting | Strengths | Typical use cases |
|-------|---------|-----------|-------------------|
| BOCPD | Online Bayesian | Sequential updates, immediate alarms | Real-time monitoring of binary/count data |
| PELT | Offline DP | Exact segmentation with linear complexity | Large offline numeric series |
| E-Divisive | Offline non-parametric | Few assumptions, multivariate | High-dimensional or unknown distributions |
| HMM/HSMM | Probabilistic state models | Regime switching with durations | Regime analysis, speech, genomics |
| SD-HMM | Duration-robust HMM | Handles over-dispersed state lengths | Compositional or proportion data |
| Within-period | Bayesian seasonal | Detects changes within cycles | Daily/weekly activity patterns |

## Installation

### From PyPI

```bash
pip install changepoint-toolkit
```

### From source

```bash
git clone https://github.com/DiogoRibeiro7/changepoint-toolkit
cd changepoint-toolkit
pip install -e .
```

Requires Python 3.10 or later.

## Basic usage

### BOCPD
```python
from changepoint_toolkit import BOCPD, BOCPDConfig, ConstantHazard
import numpy as np

rng = np.random.default_rng(0)
data = rng.binomial(1, [0.1]*50 + [0.8]*50)
model = BOCPD(ConstantHazard(mean_run_length=50), BOCPDConfig())
result = model.run(data)
print(result.change_points)
```

### PELT
```python
import numpy as np
from pelt import pelt, NormalMeanVarUnknown

y = np.r_[np.random.normal(0,1,60), np.random.normal(3,1,60)]
cost = NormalMeanVarUnknown()
res = pelt(y, cost, penalty=2*np.log(len(y)))
print(res.change_points)
```

### E-Divisive
```python
import numpy as np
from edivisive import edivisive

y = np.r_[np.random.normal(0,1,300), np.random.normal(2,1,300)]
res = edivisive(y, alpha=1.0, min_size=40, R=499, seed=0)
print(res.change_points)
```

### HMM/HSMM
```python
import numpy as np
from hsmm import HSMM, HSMMConfig, HSMMParams, PoissonDur

X = np.random.normal(0,1,(300,2))
L = np.random.normal(0,1,(300,3))  # placeholder log-likelihoods
params = HSMMParams(pi=np.full(3,1/3), A=np.full((3,3),1/3), duration=("poisson", PoissonDur(lam=np.array([30,40,50]))))
model = HSMM(HSMMConfig(K=3, Dmax=100), params)
params_fit, _ = model.fit(L)
```

### SD-HMM
```python
import numpy as np
from sdhmm import SDHMM, SDHMMConfig

X = np.random.random((500,4))
model = SDHMM(SDHMMConfig(K=3))
res = model.fit(X)
print(res.pi)
```

### Within-period changepoints
```python
import numpy as np
from within_period import WithinPeriodCPD, ModelPrior, RJConfig

rng = np.random.default_rng(1)
x = rng.binomial(1, 0.1, 240)
prior = ModelPrior(N=24, l=4)
model = WithinPeriodCPD(prior)
result = model.fit(x, RJConfig(iters=1000, burn=200, thin=5))
print(result.mode_tau)
```

## Choosing a method
- **Streaming or online monitoring** → BOCPD
- **Fast offline segmentation with cost function** → PELT
- **Minimal distributional assumptions / multivariate** → E-Divisive
- **Latent-state modeling or durations** → HMM/HSMM or SD-HMM
- **Periodic structure (e.g., daily cycles)** → Within-period CPD

## Parameter guidelines
- **BOCPD**: hazard mean run length controls sensitivity; priors tune distributional assumptions.
- **PELT**: choose penalty ~`β ≈ 2 log n` for Gaussian cost; adjust for expected number of changes.
- **E-Divisive**: `min_size` ensures segment length; `R` controls permutation replicates.
- **HMM/HSMM**: set number of states and duration distribution based on domain knowledge.
- **SD-HMM**: select state count and iteration limits; data must be non-negative and will be normalized.
- **Within-period**: period `N` and minimum segment length `l` should match the cycle of interest.

## Experimental features

The toolkit still contains a few components that are under active development and
should be considered experimental:

- **CLI wrappers** for running algorithms from the command line.  Interfaces may
  change and error handling is minimal.
- **Advanced emission models** for BOCPD and HMM/HSMM modules (e.g. Gaussian and
  Poisson alternatives) are prototypes and have not yet received the same level
  of testing as the core Beta–Bernoulli implementations.

Feedback and pull requests are welcome to help stabilise these features.

## Documentation

Detailed documentation and additional examples are available in the
[`docs/`](docs/) folder and in the [`examples/`](examples/) directory.

For planned features and long-term enhancements across modules, consult the
roadmap in [`bayesian_blocks/FUTURE_WORKS.md`](bayesian_blocks/FUTURE_WORKS.md),
which also highlights cross-module efforts like JIT acceleration and API
harmonization.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@article{ribeiro2024cpdtoolkit,
  author = {Diogo Ribeiro},
  title = {Change-Point Detection Toolkit},
  journal = {Journal of Open Source Software},
  year = {2024},
  doi = {10.21105/joss.00000}
}
```

## Acknowledgments

Developed by Diogo Ribeiro and contributors at ESMAD - Instituto Politécnico do Porto.
We thank the open-source community for foundational libraries such as NumPy.

## References
- Adams, R., & MacKay, D. (2007). *Bayesian Online Changepoint Detection*.
- Matteson, D., & James, N. (2014). *A Nonparametric Approach for Multiple Change Point Analysis of Multivariate Data*.
- Killick, R., Fearnhead, P., & Eckley, I. (2012). *Optimal Detection of Changepoints With a Linear Computational Cost*.
- Rabiner, L. (1989). *A Tutorial on Hidden Markov Models*.
- Yu, S.-Z. (2010). *Hidden Semi-Markov Models*.
- Taylor, S., Killick, R., Burr, T., & Rogerson, P. (2021). *Assessing Daily Patterns Using Within-Period Changepoint Detection*.
