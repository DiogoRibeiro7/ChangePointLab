# Environmental Science Tutorial

This tutorial demonstrates changepoint detection for climate and ecological data.

## Climate and Ecological Changepoint Detection
- Collect long‑term records (temperature, precipitation, species counts)
- Apply seasonal decomposition (STL) to remove periodicity
- Use compositing to account for spatial heterogeneity

## Seasonal Adjustment Techniques
- Use `WithinPeriodCPD` for daily/annual cycles
- De‑seasonalize with moving averages before applying PELT or BOCPD

## Detecting Anthropogenic Impacts
- **E‑Divisive** captures distributional shifts due to land‑use change
- **PELT** with variance cost highlights increased variability from climate extremes
- **SD‑HMM** models compositional shifts in biodiversity surveys

## Spatial‑Temporal Considerations
- Run detectors independently per location then cluster results
- For gridded data, apply changepoint detection after PCA/EOF reduction

## Case Study: Biodiversity Response to Climate Shifts
- Dataset: annual bird species abundance across a region for 30 years
- Goal: Detect regime shifts linked to temperature anomalies
- Method: SD‑HMM on CLR‑transformed species composition; PELT baseline on total abundance

## Parameter Tuning Considerations
- Mean run length aligned with expected ecological regime duration (5–10 years)
- Minimum segment length large enough to ignore annual noise
- Dirichlet priors in SD‑HMM set to `alpha=1` for neutrality

## Evaluation Metrics
- Rand index comparing detected regimes with known climate phases
- Change magnitude in biodiversity indices (Shannon, Simpson)
- Delay between climate anomaly and ecological response

## Interpretation Guidelines
- Validate changepoints against external drivers (ENSO events, deforestation)
- Consider confounding factors such as sampling effort

## Complete Workflow Example
See `examples/sdhmm_microbiome_analysis.py` for compositional detection techniques.

## References
- G. Rodionov, "A Sequential Algorithm for Testing Climate Regime Shifts," 2004.
- P. A. H. Parker et al., "Changepoint Detection in Ecology," 2020.
