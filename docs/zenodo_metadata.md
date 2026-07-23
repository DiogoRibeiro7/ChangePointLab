# Zenodo Metadata for ChangePointLab

## Abstract
ChangePointLab is an open-source Python library that unifies several changepoint detection approaches behind a common package interface. The current package includes Bayesian Online Change Point Detection for Bernoulli and scalar count streams, PELT-style exact penalized segmentation, energy-distance divisive algorithms, hidden Markov and semi-Markov models, state-dependent HMMs for compositional data, kernel-based methods, within-period detectors for seasonal binary signals, and sliced Poisson process detection for repeated event-time periods. Documentation, examples, tests, and command-line entry points support source-based installation and Zenodo archival releases.

## Keywords
changepoint detection; time series analysis; Bayesian online changepoint detection; PELT; energy statistics; hidden Markov model; hidden semi-Markov model; SD-HMM; within-period detection; anomaly detection; state-space models; online algorithms; offline algorithms; Python; open source; regime shift; compositional data; periodic signals; reproducible research

## Authors
- **Diogo Ribeiro** – ESMAD - Instituto Politécnico do Porto – [ORCID: 0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072) – dfr@esmad.ipp.pt

## Funding and Acknowledgments
Supported by internal research funds from ESMAD - Instituto Politécnico do Porto. The author thanks the open-source community for feedback and contributions.

## Related Publications and Datasets
- Synthetic examples included in the project repository: <https://github.com/DiogoRibeiro7/ChangePointLab>.

## Technical Specifications and System Requirements
- **Programming language:** Python ≥3.10
- **Runtime dependencies:** NumPy
- **Optional packages:** Matplotlib for plotting, pandas for CSV time-binning/data I/O
- **Dependency workflow:** Poetry-managed `pyproject.toml` and lock file
- **Operating systems:** Platform independent (tested on Linux, macOS, Windows)

## Installation
```bash
git clone https://github.com/DiogoRibeiro7/ChangePointLab
cd ChangePointLab
poetry install
```

## Usage
```python
import numpy as np
from changepoint_lab import BOCPD, ConstantHazard

data = np.random.binomial(1, 0.2, size=100)
model = BOCPD(ConstantHazard(mean_run_length=200))
result = model.run(data)
print(result.cp_prob)
```

## Citation Guidelines
If you use ChangePointLab in your research, please cite the Zenodo archive and the metadata in `CITATION.cff`:

```
Ribeiro, D. (2026). *ChangePointLab: A Unified Python Toolkit for Changepoint Detection*. Zenodo.
```

The CITATION.cff file in the repository provides bibliographic metadata for citation managers.
