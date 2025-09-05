# Zenodo Metadata for ChangePointLab

## Abstract
ChangePointLab is an open-source Python library that unifies classical and modern approaches to changepoint detection. It implements Bayesian Online Change Point Detection, Pruned Exact Linear Time segmentation, energy-distance divisive algorithms, Hidden Markov and semi-Markov models, state-dependent HMMs for compositional data, and within-period detectors for seasonal signals. A consistent API, lightweight NumPy dependency, and extensive validation utilities allow researchers to prototype, compare, and combine methods across binary, continuous, multivariate, periodic, and compositional time series. The toolkit ships with synthetic data generators, evaluation metrics, visualization helpers, and a command-line interface, promoting reproducible experimentation and cross-method insight. Detailed parameter guides, interoperability examples, and comprehensive tests support both research and teaching. ChangePointLab serves statisticians, machine-learning practitioners, and domain scientists working in fields such as IoT analytics, healthcare monitoring, finance, industrial diagnostics, and environmental science. By consolidating disparate paradigms and encouraging hybrid workflows, the project lowers the barrier to rigorous changepoint analysis and accelerates methodological innovation. Released under the permissive MIT license, ChangePointLab welcomes community contributions and is suitable for academic publication and long-term research use.

## Keywords
changepoint detection; time series analysis; Bayesian online changepoint detection; PELT; energy statistics; hidden Markov model; hidden semi-Markov model; SD-HMM; within-period detection; anomaly detection; state-space models; online algorithms; offline algorithms; Python; open source; regime shift; compositional data; periodic signals; reproducible research

## Authors
- **Diogo Ribeiro** – ESMAD - Instituto Politécnico do Porto – [ORCID: 0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072) – dfr@esmad.ipp.pt

## Funding and Acknowledgments
Supported by internal research funds from ESMAD - Instituto Politécnico do Porto. The author thanks the open-source community for feedback and contributions.

## Related Publications and Datasets
- Ribeiro, D. (2025). *Change-Point Detection Toolkit*. Journal of Open Source Software.
- Synthetic datasets and examples included in the project repository: <https://github.com/DiogoRibeiro7/articles_future>.

## Technical Specifications and System Requirements
- **Programming language:** Python ≥3.10
- **Runtime dependencies:** NumPy ≥1.21
- **Optional packages:** Matplotlib for plotting
- **Operating systems:** Platform independent (tested on Linux, macOS, Windows)

## Installation
```bash
pip install cp-ss-toolkit
```

## Usage
```python
import numpy as np
from changepoint_lab.algorithms.bayesian.bocpd import BOCPD, ConstantHazard

data = np.random.randn(100)
model = BOCPD(ConstantHazard(mean_run_length=200))
result = model.run(data)
print(result.changepoints)
```

## Citation Guidelines
If you use ChangePointLab in your research, please cite the accompanying JOSS paper and reference the Zenodo archive:

```
Ribeiro, D. (2025). Change-Point Detection Toolkit. Journal of Open Source Software. https://doi.org/10.21105/joss.00000
```

The CITATION.cff file in the repository provides bibliographic metadata for citation managers.
