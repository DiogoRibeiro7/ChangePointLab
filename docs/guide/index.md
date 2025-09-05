# Changepoint Detection Toolkit Guide

Changepoints are moments where the statistical properties of a sequence shift. Detecting them allows analysts to identify regime changes, anomalies, or structural breaks in time series and sequential data.

## Why Detect Changepoints?
- Reveal system failures or policy shifts.
- Improve forecasting by segmenting different dynamics.
- Support real‑time monitoring and alerting.

## Online vs. Offline Detection
- **Online** methods (e.g., BOCPD, within‑period CPD) update predictions sequentially and are suitable for streaming data.
- **Offline** methods (e.g., PELT, E‑Divisive) operate on complete datasets to find globally optimal segmentations.

## Bayesian vs. Frequentist Approaches
- **Bayesian** techniques model uncertainty and incorporate prior knowledge through probability distributions.
- **Frequentist** techniques focus on optimization or permutation testing to derive changepoints without priors.

## Using This Guide
This guide introduces core concepts and directs you to method‑specific tutorials:
- [Choosing a Method](choosing_methods.md)
- [Bayesian Methods](bayesian_methods.md)
- [Optimization Methods](optimization_methods.md)
- [Nonparametric Methods](nonparametric_methods.md)
- [State‑Space Methods](state_space_methods.md)
- [Visualization](visualization.md)
- [Architecture](../architecture/index.md)
- [Extending ChangePointLab](extending.md)

Mathematical notation uses standard LaTeX syntax (e.g., $p(r_t \mid x_{1:t})$ for the run‑length posterior).

For upcoming improvements and cross-module plans (e.g., Numba acceleration,
CLI utilities), see the project roadmap in the repository `CHANGELOG.md`.

