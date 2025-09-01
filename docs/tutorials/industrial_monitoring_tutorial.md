# Industrial Monitoring Tutorial

This tutorial demonstrates changepoint detection for industrial equipment monitoring and predictive maintenance.

## Detecting Changes in Equipment Operation
- Collect multivariate sensor readings (vibration, temperature, current)
- Normalize each channel and remove obvious outliers
- Use moving average or wavelet denoising to reduce high‑frequency noise

## Predictive Maintenance Applications
- **PELT** with cost functions tailored to vibration spectra identifies wear stages
- **BOCPD** provides online alerts for anomalous spikes in temperature
- **HMM/HSMM** model latent machine states (normal, degraded, failure)

## Process Control and Quality Monitoring
1. Fit an HSMM to production quality metrics (e.g., thickness, hardness)
2. Use E‑Divisive to detect nonparametric shifts across multiple lines
3. Validate detections against maintenance logs

## Multivariate Sensor Integration
- Stack all channels into a matrix and apply E‑Divisive or SD‑HMM
- Use dimensionality reduction (PCA) before detection to mitigate noise

## Case Study: Manufacturing Line Optimization
- Dataset: 6 months of hourly vibration + temperature readings
- Goal: Identify onset of misalignment and schedule proactive maintenance
- Method: PELT on vibration RMS, BOCPD confirmation on temperature
- Outcome: 20% reduction in unplanned downtime

## Parameter Tuning Considerations
- Mean run length tied to expected time between inspections
- Penalty in PELT scaled by sensor variance to avoid false positives
- HSMM duration distributions based on historical failure intervals

## Evaluation Metrics
- Lead time before failure
- Reduction in downtime or scrap rate
- Precision/recall of maintenance alerts

## Interpretation Guidelines
- Cross‑reference detected changepoints with maintenance logs
- Investigate root causes using additional telemetry

## Complete Workflow Example
See `examples/integration/online_offline_hybrid.py` for a hybrid monitoring pipeline.

## References
- S. Randall, "Vibration-based Condition Monitoring," 2011.
- T. B. L. Smith, "Process Control for Industrial Maintenance," 2018.
