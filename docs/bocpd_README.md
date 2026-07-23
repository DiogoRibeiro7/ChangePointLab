# BOCPD: Bayesian Online Changepoint Detection for Binary Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/1046174252.svg)](https://zenodo.org/badge/latestdoi/1046174252)

## Purpose

BOCPD is a Python implementation of Bayesian Online Changepoint Detection for binary (Bernoulli) data streams. The package adapts the algorithm from Adams and MacKay (2007) and includes hazard functions that can encode domain knowledge about expected changepoint locations.

Key innovations:
- **Flexible hazard functions** for periodic patterns and known boundaries
- **DST-safe binning** for proper handling of timezone transitions
- **Numerically stable** algorithms for long sequences
- **Beta-Bernoulli likelihood** for binary streams
- **Comprehensive visualization** tools and metrics

## Installation

### From Source
```bash
# Clone the repository
git clone https://github.com/DiogoRibeiro7/ChangePointLab.git
cd ChangePointLab

poetry install --with dev,docs --extras "plot data"
```

## Dependencies
- NumPy
- Optional: Matplotlib for plots
- Optional: pandas for CSV time-binning

## Basic Usage

### Simple Example
```python
import numpy as np
from changepoint_lab import BOCPD, ConstantHazard

# Create synthetic data with a changepoint
x1 = np.random.binomial(1, 0.1, size=50)  # Low probability
x2 = np.random.binomial(1, 0.8, size=50)  # High probability
x = np.concatenate([x1, x2])

# Create model with constant hazard
hazard = ConstantHazard(mean_run_length=50.0)
model = BOCPD(hazard)

# Process the data
result = model.run(x)

# Access results
print(f"CP probability at t=50: {result.cp_prob[50]:.4f}")
print(f"MAP run length at t=60: {result.map_run_length[60]}")
```

### Custom Hazard Functions
```python
from changepoint_lab import BoostedBoundaryHazard, ConstantHazard

# Boost hazard at day boundaries (every 96 points for 15-min bins)
base_hazard = ConstantHazard(mean_run_length=200.0)
boosted_hazard = BoostedBoundaryHazard(
    base=base_hazard, 
    period=96, 
    boundary_indices=frozenset([0]),  # Boost at t % 96 == 0
    boost_factor=10.0
)

model = BOCPD(boosted_hazard)
```

### Command-Line Interface
```bash
bocpd-cli --csv events.csv --bin-minutes 15 --mean-rl 96 --cp-threshold 0.6
```

For built-in demo:
```bash
bocpd-cli --demo --days 14 --period 96
```

## Documentation

- [Full Documentation](https://github.com/DiogoRibeiro7/ChangePointLab/tree/main/docs)
- [Parameter Selection Guide](https://github.com/DiogoRibeiro7/ChangePointLab/blob/main/docs/parameters/bocpd_parameters.md)
- [API Reference](https://github.com/DiogoRibeiro7/ChangePointLab/blob/main/docs/api/bocpd.rst)
- [Examples](https://github.com/DiogoRibeiro7/ChangePointLab/tree/main/examples)
- [CLI Reference](https://github.com/DiogoRibeiro7/ChangePointLab/blob/main/src/changepoint_lab/cli/bocpd_cli.py)

## How to Cite

If you use BOCPD in your research, please cite the archived ChangePointLab release:

```bibtex
@software{Ribeiro2026ChangePointLab,
  title={ChangePointLab: A Unified Python Toolkit for Changepoint Detection},
  author={Ribeiro, Diogo},
  year={2026},
  publisher={Zenodo},
  note={See CITATION.cff and the Zenodo archive for release-specific metadata}
}
```

You can also use the citation provided by the CITATION.cff file in this repository.

## Contributing

We welcome contributions to BOCPD! Please see our [contributing guidelines](CONTRIBUTING.md) for details on how to get started.

Key areas for contributions:
- Implementing additional likelihood models with oracle tests
- Adding new hazard functions
- Improving visualization tools
- Enhancing documentation and examples
- Optimizing performance

## License

BOCPD is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This package builds upon the theoretical foundation laid by Adams and MacKay (2007) in their seminal paper on Bayesian online changepoint detection.

## References

- Adams, R. P., & MacKay, D. J. (2007). Bayesian online changepoint detection. arXiv preprint arXiv:0710.3742.
- Fearnhead, P., & Liu, Z. (2007). On-line inference for multiple changepoint problems. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 69(4), 589-605.

