# BOCPD Diagram

Below is a suggested diagram to include in the JOSS paper. This would help readers understand the BOCPD algorithm's core components and workflow.

## Diagram Description

The diagram should illustrate:

1. The recursive Bayesian inference process
2. The relationship between hazard functions and run-length distribution
3. The effects of different hazard functions on changepoint detection

## Suggested Layout

```
┌────────────────────────────────────────────────────────────────┐
│                  Bayesian Online Changepoint Detection          │
└────────────────────────────────────────────────────────────────┘
                               │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌──────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Run Length     │  │    Likelihood    │  │     Hazard      │
│   Distribution   │  │      Model       │  │    Function     │
└──────────────────┘  └─────────────────┘  └─────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
              ┌─────────────────────────────┐
              │      Posterior Update       │
              └─────────────────────────────┘
                              │
                              ▼
              ┌─────────────────────────────┐
              │   Changepoint Probability   │
              └─────────────────────────────┘
```

## Example Visualization

The paper should also include a visualization showing:

1. A binary time series with true changepoints
2. The run-length posterior heatmap
3. The changepoint probability with different hazard functions

Example visualization layout:

```
Time Series with True Changepoints
[Binary data plot with vertical lines at true changepoints]

Run-Length Posterior Heatmap
[Heatmap showing P(r_t|x_{1:t}) with bright spots at changepoints]

Changepoint Probability: P(r_t=0|x_{1:t})
[Three lines showing CP probability with different hazard functions:
 - Constant hazard
 - Scheduled hazard
 - Boundary-boosted hazard]
```

The visualization should demonstrate how the boundary-boosted hazard improves detection at known boundaries compared to the constant hazard baseline.
