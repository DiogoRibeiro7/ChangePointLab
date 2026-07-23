# IoT and Sensor Data Tutorial

This tutorial demonstrates how to apply the changepoint detection toolkit to Internet of Things (IoT) sensor streams. We work through preprocessing, method selection, and evaluation on a smart‑home occupancy dataset.

## Preprocessing IoT Time Series Data
- Synchronize multi‑rate sensors with resampling (e.g., 1‑minute bins)
- Impute short gaps using forward fill and longer gaps with model‑based interpolation
- Normalize each channel to account for device‑specific scales

```python
import pandas as pd
raw = pd.read_csv("iot_home.csv", parse_dates=["timestamp"], index_col="timestamp")
series = raw.resample("1min").mean().interpolate("time").ffill()
```

## Handling Missing Values and Sensor Noise
- Use median filtering or Kalman smoothing to suppress noise
- Flag intervals with excessive missingness and exclude them from analysis

## Detecting Activity Patterns and Anomalies
- **BOCPD** with a `ScheduledHazard` to capture daily routines
- **BOCPD** with `PoissonGamma` for scalar event counts such as door openings
- **PELT** with `BetaBinomialCost` for offline binary event indicators
- **E‑Divisive** for multivariate bursts across sensors

## Multi‑Sensor Fusion for Changepoint Detection
1. Run univariate detectors per channel (e.g., motion, power)
2. Merge changepoints via clustering to obtain consensus events
3. Feed clustered events into an HMM to infer latent household states

## Case Study: Smart Home Occupancy Detection
1. Train BOCPD on aggregated motion + power usage
2. Extract changepoints and derive occupancy intervals
3. Validate against ground‑truth occupancy labels with F1 and mean delay

## Parameter Tuning Considerations
- Typical mean run length: 30–60 minutes for room‑level activity
- Use Beta‑Binomial priors for binary motion sensors, Gaussian for power data
- Minimum segment length of 5 samples avoids spurious flicker

## Evaluation Metrics
- Precision/recall on detected occupancy transitions
- Power usage reduction during unoccupied periods
- False alarm rate for anomaly detection

## Interpretation Guidelines
- Persistent changepoints near daily boundaries may indicate schedule changes
- High sensor agreement strengthens confidence in detected events

## Complete Workflow Example
A full end‑to‑end script is available in `examples/bocpd_activity_monitoring.py`.

## References
- B. Ur et al., "Smart Home Data Set," CMU, 2015.
- R. Adams & D. MacKay, "Bayesian Online Changepoint Detection," 2007.
