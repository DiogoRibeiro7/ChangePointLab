---
title: 'BOCPD: Bayesian Online Changepoint Detection for Time Series with Flexible Hazard Functions'
tags:
  - Python
  - Bayesian inference
  - changepoint detection
  - time series analysis
  - online learning
  - binary data
authors:
  - name: Diogo Ribeiro
    orcid: 0009-0001-2022-7072
    affiliation: 1
affiliations:
  - name: ESMAD - Instituto Politécnico do Porto, Portugal
    index: 1
date: 23 July 2026
bibliography: paper.bib
---

# Summary

Changepoint detection is a fundamental task in time series analysis, critical for identifying structural shifts in data streams. `BOCPD` implements an efficient, numerically stable Python package for Bayesian Online Changepoint Detection (BOCPD), specifically optimized for binary (Bernoulli) data streams with extensions for other distributions. The implementation builds upon the methodology introduced by @adams2007bayesian but adds novel hazard functions that incorporate domain knowledge about expected changepoint locations, improving detection performance in real-world applications. The package includes comprehensive visualization tools, a command-line interface, and features for handling time-of-day patterns and daylight saving time transitions in timestamp data.

# Statement of Need

Time series data often exhibits structural changes that divide the sequence into segments with distinct statistical properties. Detecting these changepoints is crucial for understanding system dynamics, identifying anomalies, and forecasting future behavior. While numerous changepoint detection methods exist, most require offline batch processing of the entire sequence, limiting their utility for streaming data applications. Furthermore, many implementations lack specific optimizations for binary data streams, which are common in IoT sensors, user behavior tracking, and event monitoring.

`BOCPD` addresses these limitations by providing:

1. An efficient online algorithm that processes data points one at a time with constant memory usage, suitable for streaming applications
2. Optimized implementation for binary (Bernoulli) data with principled Bayesian inference
3. Novel hazard function extensions that incorporate domain knowledge about when changepoints are likely to occur
4. Proper handling of timezone transitions and wall-clock time binning for real-world event data
5. A flexible API and command-line interface for both interactive analysis and automated processing pipelines

While the original BOCPD algorithm [@adams2007bayesian] provides a powerful Bayesian framework for online changepoint detection, existing implementations typically use a constant hazard function that assumes changepoints can occur anywhere with equal probability. In practice, many time series exhibit periodicity where changes are more likely at specific boundaries, such as day transitions or business hours. Our implementation extends the algorithm with scheduled and boundary-boosted hazard functions that significantly improve detection performance in such scenarios.

# Background and Theory

The Bayesian Online Changepoint Detection algorithm models a time series as a sequence of non-overlapping segments, each with stationary statistical properties. The run length $r_t$ represents the number of observations since the last changepoint at time $t$. When a new changepoint occurs, $r_t = 0$.

For binary data, we use the conjugate Beta-Bernoulli model. Given a segment with run length $r$, the posterior over the Bernoulli parameter $\theta$ after observing $s$ successes is:

$$p(\theta | r, s) = \textrm{Beta}(\theta | \alpha_0 + s, \beta_0 + r - s)$$

where $\alpha_0$ and $\beta_0$ are prior hyperparameters.

The algorithm recursively updates the probability distribution over the run length:

$$P(r_t | x_{1:t}) \propto P(r_t, x_{1:t})$$

This joint probability decomposes into:

$$P(r_t, x_{1:t}) = \sum_{r_{t-1}} P(r_t | r_{t-1}) P(x_t | r_{t-1}, x_{1:t-1}) P(r_{t-1}, x_{1:t-1})$$

The hazard function $H(r)$ controls the prior probability of a changepoint:

$$P(r_t | r_{t-1}) = \begin{cases}
H(r_{t-1}) & \textrm{if } r_t = 0 \\
1 - H(r_{t-1}) & \textrm{if } r_t = r_{t-1} + 1 \\
0 & \textrm{otherwise}
\end{cases}$$

Our implementation extends this model with hazard functions that depend on both run length and time:

1. **Constant Hazard**: $H(r, t) = 1/\lambda$ where $\lambda$ is the expected segment length
2. **Scheduled Hazard**: $H(r, t) = h_{\textrm{schedule}[t \bmod \textrm{period}]}$ for time-dependent hazards
3. **Boundary-Boosted Hazard**: $H(r, t) = \min(b \cdot H_{\textrm{base}}(r, t), 1-\epsilon)$ when $t \bmod \textrm{period} \in \textrm{boundaries}$

The changepoint probability at time $t$ is:

$$P(\textrm{changepoint at } t | x_{1:t}) = P(r_t = 0 | x_{1:t})$$

Our implementation ensures numerical stability through pruning techniques, renormalization safeguards, and efficient memory management for long sequences.

# Implementation and Features

`BOCPD` is implemented in Python with NumPy for efficient array operations. The core algorithm is designed with a modular architecture that separates the changepoint detection logic from the hazard function and likelihood model, allowing for easy extension to different data types and domain-specific hazard functions.

Key features include:

**Flexible Hazard Functions:**
- `ConstantHazard`: Traditional memoryless hazard with fixed rate
- `ScheduledHazard`: Time-dependent hazard for periodic patterns
- `BoostedBoundaryHazard`: Enhanced detection at known boundaries (e.g., day transitions)

**Pluggable Likelihood Models:**
- `BetaBernoulli`: Fully implemented model for binary data
- `PoissonGamma`: Skeleton for count data
- `GaussianNIW`: Skeleton for continuous data with Normal-Inverse-Wishart prior

**Numerical Stability:**
- Robust underflow protection with rescaling
- Efficient pruning of negligible run-length states
- Configurable top-K approximation for very long sequences

**Time Series Utilities:**
- DST-safe binning of timestamp data
- Proper handling of timezone transitions
- Flexible aggregation of events into binary indicators

**Visualization and Analysis:**
- Run-length posterior heatmaps
- Changepoint probability plots
- Comprehensive parameter selection guide

**Command-Line Interface:**
```bash
python -m bocpd_cli --csv events.csv --bin-minutes 15 --mean-rl 96 --cp-threshold 0.6
```

The package is designed to be both user-friendly for beginners and flexible for advanced users. It includes comprehensive documentation, a parameter selection guide, and extensive test coverage to ensure reliability.

# Example Usage

The following example demonstrates detection of a change in a binary sequence:

```python
import numpy as np
from bocpd import BOCPD, ConstantHazard

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
print(f"Predictive mean at t=30: {result.pred_mean[30]:.4f}")
```

For time-of-day patterns with known boundaries:

```python
from bocpd import BoostedBoundaryHazard

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

The package also provides a CLI for processing CSV files with timestamped events:

```bash
python -m bocpd_cli --csv events.csv --timestamp-col event_time \
    --bin-minutes 15 --mean-rl 96 --boost-boundary 0 \
    --period 96 --boost-factor 5.0 --timezone Europe/Lisbon
```

# Conclusion and Future Work

`BOCPD` provides a robust, efficient implementation of Bayesian Online Changepoint Detection with novel hazard function extensions for improved detection of periodic patterns. The package is designed to be accessible to researchers and practitioners in various domains, including IoT analytics, behavioral analysis, and system monitoring.

Future development plans include:
- Full implementation of additional likelihood models (Poisson-Gamma, Gaussian-NIW)
- GPU acceleration for high-throughput processing
- Online learning of hazard function parameters
- Integration with streaming data frameworks

By combining the theoretical rigor of Bayesian inference with practical optimizations for real-world use cases, `BOCPD` aims to be a valuable tool for time series analysis across various domains.

# Acknowledgements

We acknowledge contributions from the open-source community and feedback from early users who helped refine the package's API and features.

# References
