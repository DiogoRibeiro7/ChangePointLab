---
title: "ChangePointLab: A Unified Toolkit for Comprehensive Changepoint Detection"
tags:
  - changepoint
  - time-series
  - Bayesian
  - dynamic programming
  - state-space models
authors:
  - name: Diogo Ribeiro
    orcid: 0009-0001-2022-7072
    affiliation: 1
    email: dfr@esmad.ipp.pt
affiliations:
  - index: 1
    name: ESMAD – Instituto Politécnico do Porto
date: 2024-06-20
bibliography: paper.bib
---

# Summary
ChangePointLab is an open-source Python toolkit that unifies classical and modern changepoint detectors within a single, interoperable framework. The library implements Bayesian Online Change Point Detection (BOCPD) [@Adams2007], Pruned Exact Linear Time segmentation (PELT) [@Killick2012], E-Divisive energy statistics [@Matteson2013], hidden and semi-Markov models (HMM/HSMM) [@Rabiner1989; @Yu2010], a compositional SD-HMM variant [@Johnson2013], and within-period detectors for seasonal signals [@Taylor2021]. Novel hazard formulations and explicit-duration models enable analysis of binary, continuous, and compositional data. Synthetic benchmarks bundled with the project show linear-time PELT and E-Divisive achieving F1 scores above 0.9 on 10⁴‑point sequences while BOCPD’s pruning reduces memory usage by over 80 %. A uniform API, reusable evaluation utilities, and rich examples allow analysts to compare assumptions, chain algorithms, and reproduce published workflows. Tutorials, parameter guides, and decision trees make the toolkit suitable for teaching and for deployment in resource‑constrained environments. By lowering the barrier to rigorous changepoint analysis and documenting cross-method workflows, ChangePointLab accelerates methodological research and enables reproducible experimentation across domains such as IoT, finance, healthcare, industrial monitoring, and environmental science.

Installation instructions and a comprehensive user guide are available in the project README and at <https://changepointlab.readthedocs.io>.

# Statement of Need
Structural shifts in time series often signal system failures, regime transitions, or policy effects, yet the research landscape remains fragmented: individual libraries typically expose only one algorithm and assume specific data types or sampling regimes. Popular packages such as `ruptures` [@Truong2018] and `changepoint` [@Killick2014] focus on specific paradigms, requiring researchers to juggle incompatible interfaces when comparing methods. ChangePointLab provides a coherent environment where online and offline detectors, parametric and nonparametric models, and univariate and multivariate routines can be compared, combined, and tuned with minimal overhead. By supplying synthetic generators, standardized result objects, and shared evaluation metrics, the toolkit enables rigorous methodological comparisons and supports the development of new changepoint techniques and hybrid workflows.

# Software Description
## Architecture
ChangePointLab is organized into modular subpackages—`bocpd`, `pelt`, `edivisive`, `hsmm`, `sdhmm`, and `within_period`—each exposing configuration dataclasses, result containers, and plotting helpers through a top-level API. Shared utilities handle data validation, synthetic-data generation, performance evaluation, and visualization, while interoperability helpers convert outputs between methods.

## Key Features
- **BOCPD**: Custom hazard functions, run-length truncation, and posterior pruning enable real-time detection with bounded memory.
- **PELT**: Multiple cost functions (Gaussian, Poisson, Binomial) with AIC/BIC penalties and pruning achieve linear-time segmentation.
- **E-Divisive**: Energy-distance metrics, permutation tests, and vectorized prefix sums deliver scalable nonparametric detection.
- **HMM/HSMM**: Explicit-duration modeling with diverse emission distributions and Viterbi/forward‑backward routines supports state-space analysis.
- **SD-HMM & Within-Period**: RJMCMC sampling and compositional data support extend changepoint analysis to microbiome and seasonal domains.
- **Interoperability**: Examples and tests demonstrate method chaining, ensemble detection, and data-format conversions for reproducible workflows.

## Performance and Reproducibility
Benchmark tests (`tests/unit/test_performance.py`) synthesize binary, continuous, periodic, and compositional sequences with known changepoints to measure precision, recall, delay, and runtime. These tests verify that PELT and E-Divisive maintain F1 ≥ 0.9 on clean signals, while BOCPD’s pruning bounds detection delay below 5 samples on average. All examples and benchmarks rely solely on NumPy, ensuring deterministic, easily reproducible experiments.

# Impact and Audience
ChangePointLab serves statisticians, machine-learning researchers, and domain specialists who need robust, explainable changepoint analysis. The library’s ability to juxtapose and combine detectors allows researchers to explore methodological trade-offs, quantify uncertainty, and validate results via ensembles. Tutorials, parameter-selection guides, and interactive examples make the toolkit an effective teaching resource for time-series courses. Interdisciplinary applications span smart‑home occupancy inference, patient monitoring, market-regime tracking, manufacturing diagnostics, and climate studies. By packaging tests, synthetic datasets, and citation metadata, the project promotes reproducible research and open-science practices.

# Acknowledgments
I thank the contributors and open-source communities whose feedback and libraries (especially NumPy and Matplotlib) underpin this project. I also acknowledge ESMAD – Instituto Politécnico do Porto for institutional support.

# References

