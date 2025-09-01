# Bayesian Methods: BOCPD and Within-Period Detection

Bayesian changepoint detection treats unknown quantities as random variables and updates beliefs as data arrive.

## Bayesian Inference Principles
Given observations $x_{1:t}$, we maintain a posterior over the run length $r_t$:
$$p(r_t \mid x_{1:t}) \propto p(x_t \mid r_t, x_{1:t-1}) \sum_{r_{t-1}} p(r_t \mid r_{t-1}) p(r_{t-1} \mid x_{1:t-1}).$$
This recursive update yields online predictions and uncertainty estimates.

## Prior Selection and Sensitivity
- **Conjugate priors** (e.g., Beta-Bernoulli) allow closed-form updates.
- Choose hyperparameters $(\alpha,\beta)$ to encode baseline rates; extreme values may slow adaptation.
- Sensitivity analysis helps gauge robustness to prior misspecification.

## Posterior Sampling and Interpretation
Posterior run-length distributions offer:
- **MAP run length**: most probable segment length.
- **Changepoint probability**: $p(r_t = 0 \mid x_{1:t})$ signals a shift.
Sampling full trajectories provides uncertainty intervals for changepoint locations.

## Hazard Functions
The hazard $H(r)$ gives the prior probability of a changepoint at run length $r$.
- **ConstantHazard**: $H(r)=1/\lambda$ assumes memoryless arrivals.
- **ScheduledHazard**: predefines potential changepoint times $\tau_i$.
- **BoostedBoundaryHazard**: amplifies probability near boundaries or periodic indices.

In within-period detection, hazards reset every known period $P$, enabling seasonal changepoint modeling.

Refer to [BOCPD API](../api/bocpd.rst) and [Within-Period API](../api/within_period.rst) for implementation details.
