# Healthcare Data Tutorial

This tutorial applies changepoint detection methods to clinical time series for patient monitoring and early warning.

## Working with Clinical Time Series
- Align measurements to a common timeline using interpolation or kernel smoothing
- Handle irregular sampling by storing timestamps explicitly and using time‑aware models
- De‑identify data and comply with HIPAA/GDPR when sharing examples

## Detecting Physiological State Changes
- **HSMM** for modeling patient states with explicit duration distributions
- **BOCPD** for online detection of sudden deterioration (e.g., sepsis onset)
- **PELT** with Gaussian cost to segment long ICU traces

## Patient Monitoring and Early Warning Systems
1. Train a multivariate HSMM on vitals (heart rate, BP, SpO₂)
2. Use BOCPD to flag abrupt changes in qSOFA score
3. Combine alerts with clinical rules to reduce false positives

## Handling Irregularly Sampled Medical Data
- Use continuous‑time BOCPD variants or resample to regular intervals with imputation
- Apply Kalman filters to smooth noisy measurements prior to detection

## Case Study: Monitoring Disease Progression
- Dataset: 30‑day heart‑rate and activity logs for a cardiac patient
- Goal: Detect transitions between rest, moderate activity, and tachycardia episodes
- Method: HSMM with 3 states; PELT baseline for comparison

## Parameter Tuning Considerations
- State counts reflect clinical knowledge (e.g., normal, warning, critical)
- Prior hyperparameters for BOCPD derived from historical patient data
- Minimum segment length tied to physiologic response times (e.g., 5 minutes)

## Evaluation Metrics
- Event detection F1 against annotated episodes
- Time to detection for early warning
- Clinical utility: reduction in false alarms per day

## Interpretation Guidelines
- Cross‑validate against clinician‑labeled events before deployment
- Combine changepoints with context (medications, interventions) for causality

## Complete Workflow Example
Example code is provided in `examples/hsmm_medical_monitoring.py`.

## References
- P. Schulam & S. Saria, "A Framework for Individualized Disease Trajectory Prediction," 2015.
- D. Barry & J. Hartigan, "A Bayesian Analysis for Change Point Problems," 1993.
