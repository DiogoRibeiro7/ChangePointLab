# Visualizing Changepoints

Visualization aids interpretation and communication of detected changepoints.

## Standard Plots
- **BOCPD**: plot run-length posterior heatmaps and changepoint probabilities.
- **PELT**: overlay optimal segments on the data.
- **E-Divisive**: show cumulative energy statistics and segmentation points.
- **HMM/HSMM**: display state sequences and smoothed posteriors.

## Comparative Visualization
Use side-by-side subplots to compare methods on the same dataset, highlighting consensus or disagreement.

## Interactive Options
Tools like `plotly` or `bokeh` allow zooming into suspected changepoints and toggling method overlays.

## Interpretation Guidelines
- Align predicted changepoints with known events when possible.
- Consider uncertainty bands from Bayesian methods.
- Use ensemble visualizations to corroborate findings across techniques.

See [examples/multi_method_comparison.py](../../examples/multi_method_comparison.py) for reusable plotting helpers.
