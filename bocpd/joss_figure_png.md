# BOCPD Figure Description

Below is a description of the main figure to include in the JOSS paper. This figure illustrates the key concepts and advantages of the BOCPD package.

## Figure 1: BOCPD Overview and Hazard Function Comparison

This figure should be a multi-panel visualization consisting of:

### Panel A: Algorithm Overview
A flowchart depicting the BOCPD algorithm components:
- Run-length distribution updating
- Hazard function application
- Likelihood model
- Posterior inference

### Panel B: Binary Data with Changepoints
A step plot showing binary (0/1) data with vertical lines indicating true changepoints.

### Panel C: Run-Length Posterior Heatmap
A heatmap visualization of P(r_t | x_{1:t}) showing the run-length posterior over time, with brighter colors at changepoints where run length resets to 0.

### Panel D: Hazard Function Comparison
Multiple lines showing changepoint probability P(r_t=0 | x_{1:t}) for different hazard functions:
- Constant hazard (traditional BOCPD)
- Scheduled hazard with periodic boundary enhancement
- Boundary-boosted hazard

The plot should clearly demonstrate how specialized hazard functions improve detection at known boundaries compared to the constant hazard baseline, especially for periodic data with time-of-day patterns.

### Caption:
**Figure 1. BOCPD algorithm overview and hazard function comparison.** 
(A) Flowchart of the BOCPD algorithm showing recursive Bayesian inference. 
(B) Binary time series with true changepoints marked by vertical dashed lines. 
(C) Run-length posterior heatmap showing the probability distribution P(r_t | x_{1:t}) over time, with bright spots at changepoints where run length resets to 0. 
(D) Changepoint probability P(r_t=0 | x_{1:t}) for three different hazard functions, demonstrating how specialized hazard functions improve detection performance at known boundaries. The boundary-boosted hazard (green) shows stronger detection at period boundaries compared to constant hazard (blue).

## Optional Additional Figure: Parameter Sensitivity

If space allows, a second figure showing:

### Panel A: Prior Parameter Effects
Multiple lines showing changepoint probability for different Beta prior parameters (α₀, β₀):
- Jeffreys-like prior (α₀=β₀=0.5)
- Uniform prior (α₀=β₀=1.0)
- Informative prior (e.g., α₀=0.2, β₀=0.8)

### Panel B: Mean Run Length Effects
Multiple lines showing changepoint probability for different mean run length values:
- Short (λ=50)
- Medium (λ=100)
- Long (λ=200)

### Caption:
**Figure 2. Parameter sensitivity analysis.** 
(A) Effect of Beta prior parameters (α₀, β₀) on changepoint detection sensitivity. Smaller values adapt quickly to changes but may be more prone to false positives. 
(B) Effect of mean run length (λ) on detection. Shorter values increase sensitivity but may lead to false positives, while longer values reduce sensitivity but improve precision.
