---
title: 'Change-Point Detection Toolkit'
authors:
  - name: Diogo Ribeiro
    orcid: 0009-0001-2022-7072
    affiliation: 1
affiliations:
  - name: ESMAD - Instituto Politécnico do Porto
    index: 1
date: 2024
bibliography: paper.bib
---

# Summary

The *Change-Point Detection Toolkit* (CPDToolkit) is a Python library that gathers diverse algorithms for locating structural changes in temporal data. It offers both streaming and batch detection methods, harmonizes their interfaces, and supplies utilities for visualization and benchmarking. By bundling complementary approaches—Bayesian, energy‑based, kernel, and state‑space models—the toolkit lowers the barrier to experimenting with and comparing modern change-point techniques.

# Statement of need

Research and practice in change-point detection often require trying several algorithms to balance statistical power, computational cost, and modeling assumptions. Implementations are scattered across projects with incompatible APIs and output formats, complicating empirical comparison and hindering reuse. CPDToolkit provides a single dependency‑free interface that unifies popular algorithms under common data structures. This design streamlines benchmarking, facilitates method selection for practitioners, and serves as a foundation for teaching and reproducible research.

# Mathematical background and implemented methods

## Bayesian Online Change-Point Detection (BOCPD)

BOCPD infers the probability of a regime shift at each observation using Bayesian sequential analysis [@Adams2007]. A hazard function models the prior over run lengths, and conjugate exponential-family likelihoods permit efficient updates. The toolkit exposes flexible hazard functions, posterior sampling, and visualization of run-length posteriors.

## E-Divisive energy-based detection

E-Divisive identifies distributional changes by maximizing an energy distance between segments [@Matteson2013]. The method performs a hierarchical divisive search and assesses significance via permutations. CPDToolkit implements the algorithm for multivariate series and returns both estimated change points and associated p‑values.

## Pruned Exact Linear Time (PELT)

PELT solves the optimal partitioning problem with dynamic programming and a pruning rule that yields expected linear complexity [@Killick2012]. The implementation accommodates user‑defined cost functions and penalties, enabling rapid multiple change-point detection in long univariate signals.

## Hidden Markov and Hidden Semi-Markov Models

Hidden Markov Models describe sequences through latent states with memoryless transitions [@Rabiner1989]. Hidden Semi-Markov Models extend this by explicitly modeling state durations [@Yu2010]. CPDToolkit provides maximum-likelihood estimation for both formulations with Gaussian emissions and returns decoded state sequences and durations.

## Scaled-Dirichlet HMM (SD-HMM)

SD-HMM generalizes HSMMs with flexible, nonparametric duration distributions based on scaled Dirichlet priors [@Johnson2013]. The package supports both single and mixture components, enabling rich dwell-time modeling for complex event streams.

## Within-period changepoint detection

Within-period methods assume periodic binary observations and apply reversible-jump MCMC to locate change points within each cycle [@Taylor2021]. The implementation produces posterior samples of change locations and supports parallel tempering for efficient exploration.

Figure @fig-overview provides a high-level flowchart contrasting the conceptual focus of each algorithm in the toolkit.

![Conceptual differences among included methods](paper/figures/methods_flowchart.png){#fig-overview}

# Implementation and key features

All algorithms share NumPy array inputs and return typed `ChangePointResult` objects, simplifying downstream analysis. The `AlgorithmRegistry` unifies configuration of heterogeneous methods, exposing metadata, parameter validation, and a common `run` function. This harmonization allows side‑by‑side evaluation of methods such as E‑Divisive, kernel change‑point detection [@Arlot2012], and state‑space models without bespoke glue code. Visualization helpers plot detected segments, posterior run-lengths, and latent state sequences.

# Example usage

```python
from toolkit.api_harmonizer import registry
import numpy as np

# synthetic signal with two shifts
np.random.seed(0)
x = np.r_[np.random.normal(0, 1, 200),
         np.random.normal(2, 1, 200)]

# compare algorithms through the unified API
ediv = registry.run("edivisive", data=x[:, None], min_size=50)
kcp  = registry.run("kcp_penalized", data=x, kernel="rbf",
                    gamma=np.log(len(x)), min_size=50)

print("E-Divisive:", ediv.change_points)
print("KCP:", kcp.change_points)
```

The same registry can dispatch to `hsmm` or `sdhmm` for state‑space inference. For online detection, the `BOCPD` class offers streaming updates of run‑length distributions.

Figure @fig-comparison shows a synthetic binary sequence alongside changepoint estimates from each method, and Figure @fig-performance reports the discrepancy in detected counts. A decision tree to guide method selection is given in Figure @fig-decision.

![Synthetic sequence analysed by multiple methods](paper/figures/comparison.png){#fig-comparison}

![Detection count error by method](paper/figures/performance.png){#fig-performance}

![Flowchart for choosing a detection method](paper/figures/decision_tree.png){#fig-decision}

# Potential applications and future work

CPDToolkit is applicable to finance, network security, environmental monitoring, and any domain requiring detection of structural breaks or latent regimes. Planned enhancements include additional cost functions for PELT, GPU‑accelerated kernels, automated hyper‑parameter selection, and higher‑level benchmarking workflows to facilitate reproducible comparisons across datasets.

# Acknowledgements

The author thanks the open-source community for discussions and contributions on change-point methodology.

# References
