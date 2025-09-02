---
title: 'ChangePointLab: A Unified Toolkit for Time Series Changepoint Detection'
tags:
  - Python
  - time series
  - changepoint detection
  - segmentation
  - Bayesian statistics
  - kernel methods
  - hidden Markov models
authors:
  - name: Diogo F. Ribeiro
    orcid: 0009-0001-2022-7072
    affiliation: 1
    email: dfr@esmad.ipp.pt
affiliations:
  - name: ESMAD, Instituto Politécnico do Porto, Portugal
    index: 1
date: 15 August 2025
bibliography: paper.bib
---

# Summary

Time series data frequently undergoes structural changes that manifest as shifts in statistical properties. Detecting these changepoints is essential across domains from finance to healthcare to environmental monitoring. ChangePointLab provides a comprehensive, unified Python framework that brings together classical and modern approaches to changepoint detection under a consistent API. The toolkit implements Bayesian online detection, optimal segmentation algorithms, nonparametric energy-based methods, and state-space models. By offering standardized interfaces, extensive visualization tools, and domain-specific utilities, ChangePointLab enables researchers and practitioners to easily compare, combine, and deploy multiple detection strategies on diverse time series data.

# Statement of Need

Abrupt changes in time series data signify critical transitions in underlying systems, whether financial market regime shifts, patient health deterioration, or climate pattern alterations. These changepoints provide valuable insights for both retrospective analysis and real-time monitoring. However, existing changepoint detection tools in Python typically implement individual algorithms in isolation, with inconsistent interfaces that make method comparison and ensemble approaches difficult. 

ChangePointLab addresses this fragmentation by providing:

1. A **unified framework** encompassing major changepoint detection paradigms
2. **Consistent APIs** that simplify method comparison and hybrid workflows
3. **Optimized implementations** that scale to large datasets
4. **Domain-specific tutorials** for finance, healthcare, IoT, and environmental applications
5. **Comprehensive visualization** tools for result interpretation
6. **Robust evaluation metrics** for quantitative assessment

This standardization significantly reduces the barrier to entry for non-specialists while providing researchers with tools to develop and benchmark novel methods. The library serves both as a practical tool for applied data analysis and a foundation for methodological research in changepoint detection.

# Core Functionality

ChangePointLab includes five primary modules:

## Bayesian Online Changepoint Detection (BOCPD)

The `bocpd` module implements Adams & MacKay's [@adams2007bayesian] sequential inference approach. It provides:

- Conjugate models for different data types (Gaussian, Poisson, Bernoulli)
- Configurable hazard functions including constant, scheduled, and boundary-boosted variants
- Memory-efficient run-length distribution tracking for streaming applications
- Probabilistic changepoint assessment with uncertainty quantification

```python
from changepoint_lab import bocpd
detector = bocpd.BOCPD(hazard=bocpd.ConstantHazard(200))
result = detector.fit(data)
plt.plot(result.changepoint_probability)
```

## Pruned Exact Linear Time (PELT)

The `pelt` module implements Killick's algorithm [@killick2012optimal] for efficient exact segmentation:

- Diverse cost functions (Gaussian, binomial, custom)
- Information-theoretic penalties (AIC, BIC)
- Linear-time complexity through dynamic pruning
- Support for concave penalties

```python
from changepoint_lab import pelt
cost_function = pelt.NormalMeanVarUnknown()
result = pelt.detect(data, cost_function, penalty=pelt.bic_penalty(2, len(data)))
```

## E-Divisive

The `edivisive` module implements Matteson's nonparametric approach [@matteson2014nonparametric] using energy statistics:

- Distribution-free detection without parametric assumptions
- Permutation testing for statistical significance
- Multivariate changepoint detection
- Block bootstrapping for dependent data

```python
from changepoint_lab import edivisive
result = edivisive.detect(data, alpha=1.0, significance=0.05)
```

## Hidden Semi-Markov Models (HSMM)

The `hsmm` module implements Yu's explicit-duration hidden semi-Markov models [@yu2010hidden]:

- Explicit modeling of state durations
- Multiple emission models (diagonal/full covariance Gaussian, autoregressive)
- EM parameter estimation
- Viterbi decoding for optimal state sequence recovery

```python
from changepoint_lab import hsmm
model = hsmm.HSMM(n_states=3, duration="poisson")
result = model.fit(data)
```

## Kernel Changepoint Detection (KCP)

The `kcp` module provides kernel-based methods for flexible non-linear segmentation:

- Multiple kernel functions (linear, RBF, custom)
- Random Fourier Features for large-scale applications
- Automatic bandwidth selection via cross-validation
- BIC-style model selection

```python
from changepoint_lab import kcp
kernel_matrix = kcp.gram_rbf(data)
result = kcp.detect_penalized(kernel_matrix, gamma=np.log(len(data)))
```

# Comparison with Existing Software

ChangePointLab builds upon and integrates functionality from several existing packages while providing unique capabilities:

| Feature | ChangePointLab | ruptures | changepy | astropy.stats |
|---------|----------------|----------|----------|---------------|
| Bayesian online detection | ✓ | ✗ | ✓ | ✗ |
| PELT algorithm | ✓ | ✗ | ✗ | ✗ |
| Binary/count/event data | ✓ | ✗ | ✗ | ✓ |
| Kernel-based methods | ✓ | ✓ | ✗ | ✗ |
| HSMM/state-space models | ✓ | ✗ | ✗ | ✗ |
| Multivariate E-Divisive | ✓ | ✓ | ✗ | ✗ |
| Auto parameter selection | ✓ | ✓ | ✗ | ✗ |
| Uncertainty quantification | ✓ | ✗ | ✓ | ✗ |
| Domain-specific tutorials | ✓ | ✗ | ✗ | ✓ |
| Unified API across methods | ✓ | ✓ | ✗ | ✗ |

While individual algorithms may be available elsewhere, ChangePointLab uniquely provides a unified framework with consistent APIs, comprehensive documentation, and optimized implementations that facilitate method comparison and ensemble approaches.

# Applications

ChangePointLab has been successfully applied across diverse domains:

- **Finance**: detecting market regime shifts and volatility changes
- **Healthcare**: identifying patient state transitions in physiological time series
- **Environmental Science**: detecting climate anomalies and ecological shifts
- **Industrial Monitoring**: predictive maintenance and quality control
- **IoT Analytics**: activity recognition and anomaly detection

The toolkit includes domain-specific tutorials and example datasets that demonstrate these applications, making domain adaptation straightforward for new users.

# Performance and Scalability

Performance optimization is a key focus, with specialized implementations for different data scales:

- **Small-to-medium datasets**: Exact methods with careful algorithm design
- **Large datasets**: Approximation techniques (RFF, grid search, pruning)
- **Streaming data**: Online algorithms with bounded memory usage

Benchmark results show that ChangePointLab achieves comparable or better performance than specialized implementations while offering greater flexibility and functionality.

# Community and Documentation

ChangePointLab emphasizes accessibility through:

- Comprehensive API documentation with mathematical details
- Interactive tutorials for each algorithm
- Domain-specific application guides
- Performance comparison benchmarks
- Contribution guidelines for extending the library

The package is released under the MIT license to encourage both academic and commercial adoption.

# Acknowledgments

We thank the open-source community, particularly the contributors to astropy.stats, ruptures, and changepy, whose work informed aspects of this implementation. Special thanks to faculty and researchers at Instituto Politécnico do Porto for their feedback and testing.

# References
