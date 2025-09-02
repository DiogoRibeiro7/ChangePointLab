# Scientific Reproducibility Plan for ChangePointLab

This document outlines our comprehensive plan to enhance the scientific reproducibility of ChangePointLab, ensuring that results can be consistently replicated across different environments and time periods.

## 1. Standardized Benchmark Datasets

### Synthetic Data Generation

We will provide fixed-seed generators for canonical changepoint detection problems:

```python
# From changepoint_lab.benchmarks.synthetic
from changepoint_lab.benchmarks import generate_mean_shift, generate_variance_change

# Generate standard mean shift benchmark (3 regimes)
X_mean, true_cps = generate_mean_shift(
    n_samples=1000,
    regime_lengths=[300, 400, 300],
    means=[0.0, 2.0, -1.0],
    noise_scale=0.5,
    seed=42
)

# Generate standard variance change benchmark (2 regimes)
X_var, true_cps = generate_variance_change(
    n_samples=1000,
    regime_lengths=[500, 500],
    scales=[0.5, 2.0],
    seed=42
)
```

#### Core Synthetic Datasets

1. **mean_shift**: Abrupt changes in mean level
2. **variance_change**: Changes in volatility with constant mean
3. **mean_variance_change**: Simultaneous mean and variance changes
4. **multivariate_shift**: Correlated multivariate series with regime changes
5. **seasonal_shift**: Seasonal data with changing parameters
6. **autoregressive_shift**: AR processes with changing coefficients
7. **binary_regime**: Binary sequences with changing probabilities
8. **heavy_tailed**: Non-Gaussian data with shifting parameters
9. **gradual_change**: Gradually evolving parameters
10. **oscillatory**: Changing frequency/amplitude oscillations

### Real-World Reference Datasets

We will maintain a curated collection of preprocessed real-world datasets with annotated changepoints:

1. **financial_crises**: Market indices during major financial events (1987-2023)
2. **climate_shifts**: Global temperature anomalies with known climate shifts
3. **eeg_seizures**: Epileptic seizure recordings with expert annotations
4. **genomic_segments**: Copy number variations with validated breakpoints
5. **speech_boundaries**: Phoneme and word boundaries in speech recordings

### Dataset Registry and Versioning

All datasets will be:
- Versioned with checksums to ensure reproducibility
- Available through a consistent API
- Documented with metadata about their properties
- Accessible through both direct download and programmatic fetching

```python
from changepoint_lab.datasets import load_dataset

# Load a specific version of a dataset
climate_data = load_dataset("climate_shifts", version="1.2.0")
```

## 2. Reproducible Examples

### Validation Notebooks

For each algorithm, we will provide Jupyter notebooks that:
- Demonstrate the algorithm on canonical problems
- Show parameter sensitivity analysis
- Compare against alternative approaches
- Include expected outputs and visualizations

All notebooks will:
- Run with fixed random seeds
- Include version information for dependencies
- Be automatically tested in CI to ensure they remain functional
- Be rendered in the documentation

### Executable Tutorials

Domain-specific tutorials will be executable and reproducible:

```python
# Example: Financial changepoint detection tutorial
from changepoint_lab.tutorials import financial

# Run the complete tutorial with reproducible output
tutorial = financial.market_regime_detection(
    data="sp500_returns",
    period="2000-2020",
    reproducible=True  # Uses fixed seeds and parameters
)

# Access key outputs
detected_regimes = tutorial.results
performance_metrics = tutorial.metrics
tutorial.visualize()  # Generate key figures
```

### Comparison Workflows

Standard comparison workflows will be available for benchmarking:

```python
from changepoint_lab.benchmarks import compare_methods

# Compare multiple methods on a standard dataset
results = compare_methods(
    dataset="mean_shift",
    methods=["pelt", "bocpd", "edivisive"],
    metrics=["f1", "ari", "runtime"],
    seeds=range(10)  # Run with 10 different seeds
)

# Generate standard comparison figures
results.plot_accuracy()
results.plot_runtime()
```

## 3. Versioned Documentation

### Version-Specific Documentation

All documentation will be versioned and tied to specific releases:

- API reference will include version tags
- Archived documentation for all previous versions
- Clear changelogs highlighting API changes
- Migration guides between major versions

### Result Reproducibility Matrix

For key algorithms, we will maintain a reproducibility matrix showing:

| Algorithm | Version | Dataset | Expected Output | Runtime Range | Memory Range |
|-----------|---------|---------|-----------------|---------------|--------------|
| PELT      | 1.0.0   | mean_shift_v1 | CP @ [300, 700] | 10-15ms | 15-20MB |
| BOCPD     | 1.0.0   | binary_regime_v1 | P(CP) > 0.8 @ [500] | 20-25ms | 25-30MB |

### Computational Environment Documentation

For each example, we will document:
- Required dependencies with version ranges
- Hardware specifications used for benchmarks
- Expected runtime variations across environments

## 4. Containerization

### Docker Images

We will provide Docker images for reproducible environments:

```bash
# Pull the ChangePointLab research environment
docker pull changepointlab/research:1.0.0

# Run a specific example in the container
docker run changepointlab/research:1.0.0 python -m changepoint_lab.examples.pelt_example
```

### Singularity Containers

For HPC environments, we will maintain Singularity definition files:

```bash
# Build the Singularity container
singularity build cpl.sif singularity/changepoint_lab.def

# Run in the container
singularity exec cpl.sif python my_analysis.py
```

### Development Containers

VS Code Dev Containers configuration will be provided for consistent development environments.

## 5. Scientific Workflow Integration

### MLflow Integration

We will provide MLflow integrations for experiment tracking:

```python
from changepoint_lab.tracking import mlflow_tracking

# Track experiment with MLflow
with mlflow_tracking("pelt_experiment"):
    # Parameters are automatically logged
    result = pelt(data, cost_fn, penalty=penalty)
    
    # Log custom metrics
    mlflow_tracking.log_metric("num_changepoints", len(result.change_points))
    
    # Log figures
    fig = plot_segments(data, result)
    mlflow_tracking.log_figure(fig, "segmentation.png")
```

### Sacred Integration

For experiment management:

```python
from changepoint_lab.tracking import sacred_experiment

@sacred_experiment
def analyze_climate_data(dataset="global_temp", method="pelt", penalty=10):
    # Analysis code here
    pass

# Run the experiment
analyze_climate_data()
```

### Weights & Biases Integration

For experiment visualization and sharing:

```python
from changepoint_lab.tracking import wandb_tracking

with wandb_tracking("changepoint_analysis"):
    # Run analysis
    results = compare_methods(...)
    
    # Create interactive visualization
    wandb_tracking.log_interactive_plot(results.create_comparison_plot())
```

## 6. Parameter Sensitivity Analysis

### Automated Sensitivity Testing

We will include tools for systematic parameter sensitivity analysis:

```python
from changepoint_lab.sensitivity import parameter_sweep

# Define parameter grid
param_grid = {
    "penalty": [0.1, 1.0, 10.0, 100.0],
    "min_size": [5, 10, 20]
}

# Run parameter sweep
sensitivity_results = parameter_sweep(
    algorithm="pelt",
    dataset="mean_shift",
    param_grid=param_grid,
    metric="f1_score",
    n_repeats=5  # Repeat each configuration 5 times
)

# Visualize parameter sensitivity
sensitivity_results.plot_heatmap()
sensitivity_results.plot_interaction_effects()
```

### Robustness Reports

For each algorithm, we will provide robustness reports showing:
- Stability to noise level variations
- Sensitivity to primary parameters
- Performance under edge cases

### Confidence Interval Estimation

For stochastic algorithms, we will provide tools for confidence interval estimation:

```python
from changepoint_lab.statistics import bootstrap_confidence

# Estimate confidence intervals on changepoint locations
confidence_intervals = bootstrap_confidence(
    algorithm="bocpd",
    data=data,
    n_bootstrap=1000,
    confidence_level=0.95
)

# Visualize with confidence bands
plot_changepoints_with_confidence(data, confidence_intervals)
```

## 7. Research Guidelines

### Reporting Standards

We will provide templates for reporting changepoint analysis results in publications:

```markdown
# Changepoint Analysis Reporting Template

## Dataset
- Name: [Dataset Name]
- Version: [Dataset Version]
- Size: [Number of observations]
- Dimensionality: [Number of variables]
- Source: [Citation or URL]

## Algorithm
- Method: [Algorithm name]
- Version: [ChangePointLab version]
- Parameters: [List of key parameters with values]
- Computational environment: [Hardware/software details]

## Results
- Number of detected changepoints: [Count]
- Locations: [List of positions]
- Uncertainty: [Confidence intervals if applicable]
- Metrics: [Performance metrics if ground truth available]

## Reproducibility Information
- Random seed: [Seed value]
- Software dependencies: [Version numbers]
- Runtime: [Execution time]
- Repository: [Link to code repository]
```

### Domain-Specific Guidance

We will provide specialized guidelines for different research domains:

#### Financial Time Series
- Recommendations for market data preprocessing
- Specialized evaluation metrics for financial regimes
- Backtest procedures for trading strategies

#### Healthcare Monitoring
- Guidelines for physiological data processing
- Patient privacy considerations
- Clinical validation approaches

#### Climate Data Analysis
- Handling of seasonal components
- Long-term trend separation
- Significance testing for climate shifts

## 8. Integration with Scientific Publishing

### Figure Reproduction

Tools for exactly reproducing publication figures:

```python
from changepoint_lab.publications import reproduce_figure

# Reproduce a specific figure from a paper
fig = reproduce_figure(
    paper_id="ribeiro2025changepoint",
    figure_number=3,
    panel="b"
)
```

### Code Generation

Automatic generation of minimal code examples for papers:

```python
from changepoint_lab.publications import generate_minimal_example

# Generate minimal reproducible example
code = generate_minimal_example(
    algorithm="pelt",
    dataset="financial_crises",
    parameters={"penalty": 10, "min_size": 20}
)

# Save to file
with open("paper_example.py", "w") as f:
    f.write(code)
```

### Paper Template

LaTeX template for papers with reproducible changepoint analysis:

```latex
\documentclass{article}
\usepackage{changepointlab}

\begin{document}

\title{My Changepoint Analysis}
\author{Author Name}
\maketitle

\begin{cpanalysis}{dataset=climate_shifts, method=pelt, parameters={penalty=10}}
% This environment automatically runs the analysis and creates figures
% when the document is compiled with the changepoint-lab-latex tool
\end{cpanalysis}

\end{document}
```

## 9. Long-term Archiving

### Zenodo Integration

Automatic deposition of code, data, and results to Zenodo:

```python
from changepoint_lab.archiving import zenodo_archive

# Archive a complete analysis
doi = zenodo_archive(
    experiment_id="climate_analysis_2025",
    title="Climate Changepoint Analysis 2025",
    authors=["Ribeiro, D."],
    description="Comprehensive analysis of climate shifts using ChangePointLab",
    license="MIT"
)

print(f"Archived with DOI: {doi}")
```

### Reproducible Research Object

Creation of standalone research objects containing:
- Code
- Data
- Environment specifications
- Results
- Documentation

```python
from changepoint_lab.archiving import create_research_object

# Create a complete research object
ro = create_research_object(
    experiment_id="financial_regimes_study",
    include_data=True,
    include_environment=True,
    include_results=True
)

# Export as a standalone package
ro.export("financial_regimes_study.zip")
```

## Conclusion

This comprehensive plan will significantly enhance the scientific reproducibility of ChangePointLab, allowing researchers to confidently build on our work and ensuring that results can be consistently replicated. By implementing these practices, we aim to set a new standard for reproducible research in the changepoint detection community.

We welcome feedback and contributions to this plan from the scientific community.
