# Bayesian Blocks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful, feature-rich Python implementation of Bayesian Blocks for optimal changepoint detection and data segmentation. This library provides state-of-the-art algorithms for detecting regime changes in time series, event data, and counting processes.

## 🚀 Features

### Core Algorithms

- **Event Times**: Optimal segmentation of unbinned Poisson processes
- **Binned Counts**: Analysis of Poisson count data with variable bin widths
- **Bernoulli Data**: Changepoint detection in binary sequences and success/failure data

### Advanced Capabilities

- **Unified API** with automatic data type detection
- **Enhanced Results** with AIC, BIC, and model diagnostics
- **Bootstrap Confidence Intervals** for statistical uncertainty quantification
- **Cross-Validation** for automatic parameter selection
- **Streaming Analysis** for real-time data processing
- **Adaptive Algorithms** that adjust parameters based on data characteristics

### Visualization

- **Professional Plotting** with matplotlib integration
- **Interactive Visualizations** with Plotly support
- **Diagnostic Plots** for model assessment
- **Comparison Tools** for analyzing multiple results

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bayesian_blocks.git
cd bayesian_blocks

# Install dependencies
pip install numpy matplotlib seaborn

# Optional: Install for interactive plotting
pip install plotly

# Optional: Install for testing
pip install pytest hypothesis
```

## 🔥 Quick Start

### Basic Usage

```python
import numpy as np
from bayesian_blocks import bayesian_blocks_counts
from bb_plotting import plot_blocks_index
import matplotlib.pyplot as plt

# Generate sample data with a changepoint
rng = np.random.default_rng(42)
rate = np.concatenate([np.full(80, 3.0), np.full(120, 0.8)])
counts = rng.poisson(rate)

# Detect changepoints
result = bayesian_blocks_counts(counts, p0=0.05)

# Plot results
ax = plot_blocks_index(N=len(counts), result=result, 
                      ylabel="rate", title="Poisson Changepoint Detection")
plt.show()

print(f"Detected {len(result.block_value)} blocks")
print(f"Changepoints at indices: {result.change_points}")
```

### Unified API with Auto-Detection

```python
from bayesian_blocks import bayesian_blocks

# Automatically detects data type and applies appropriate algorithm
binary_data = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1]
result = bayesian_blocks(binary_data, data_type="auto")  # Detects Bernoulli

count_data = [3, 5, 2, 8, 1, 1, 9, 2]
result = bayesian_blocks(count_data, data_type="auto")   # Detects counts

event_times = [1.2, 1.8, 2.1, 4.3, 4.7, 5.1]
result = bayesian_blocks(event_times, data_type="auto")  # Detects events
```

### Advanced Configuration

```python
from bayesian_blocks import bayesian_blocks, BBConfig

# Custom configuration
config = BBConfig(
    p0=0.01,              # Stricter changepoint detection
    min_block_size=5,     # Minimum 5 data points per block
    method="dp"           # Dynamic programming method
)

result = bayesian_blocks(data, data_type="counts", config=config)

# Enhanced results with model diagnostics
print(f"AIC: {result.aic:.2f}")
print(f"BIC: {result.bic:.2f}")
print(f"Log-likelihood: {result.log_likelihood:.2f}")
```

## 📚 Comprehensive Examples

### 1\. Event Time Analysis (Astronomy/Physics)

```python
from bayesian_blocks import bayesian_blocks_events
from bb_plotting import plot_blocks_time

# Simulate gamma-ray burst data
rng = np.random.default_rng(42)

# Background + burst + background
background_rate = 0.1
burst_rate = 5.0

# Generate event times
events_bg1 = np.cumsum(rng.exponential(1/background_rate, 100))
events_bg1 = events_bg1[events_bg1 < 100]

events_burst = 100 + np.cumsum(rng.exponential(1/burst_rate, 50))
events_burst = events_burst[events_burst < 120]

events_bg2 = 120 + np.cumsum(rng.exponential(1/background_rate, 100))
events_bg2 = events_bg2[events_bg2 < 200]

all_events = np.concatenate([events_bg1, events_burst, events_bg2])

# Detect rate changes
result = bayesian_blocks_events(all_events, t_start=0, t_stop=200, p0=0.01)

# Plot results
ax = plot_blocks_time(t_min=0, t_max=200, result=result, 
                     title="Gamma-Ray Burst Detection")
plt.show()
```

### 2\. Financial Time Series Analysis

```python
# Volatility regime detection
returns = np.concatenate([
    rng.normal(0, 0.01, 100),  # Low volatility
    rng.normal(0, 0.05, 50),   # High volatility  
    rng.normal(0, 0.01, 100)   # Return to low volatility
])

# Use absolute returns as proxy for volatility
volatility_proxy = np.abs(returns)

result = bayesian_blocks_counts(volatility_proxy, p0=0.05)
print(f"Detected {len(result.block_value)} volatility regimes")
```

### 3\. A/B Testing and Conversion Analysis

```python
# Simulate A/B test with conversion rate change
n_trials = np.ones(200, dtype=int)  # 1 trial per observation
conversions_before = rng.binomial(n_trials[:100], 0.15)  # 15% conversion rate
conversions_after = rng.binomial(n_trials[100:], 0.22)   # 22% conversion rate
all_conversions = np.concatenate([conversions_before, conversions_after])

result = bayesian_blocks_bernoulli(all_conversions, n_trials, p0=0.05)

print(f"Conversion rates by period:")
for i, rate in enumerate(result.block_value):
    start_idx = int(result.edges[i])
    end_idx = int(result.edges[i+1])
    print(f"  Period {i+1} (days {start_idx}-{end_idx}): {rate:.1%}")
```

## 🎨 Advanced Visualization

### Diagnostic Plots

```python
from bb_plotting import BBPlotter

# Create comprehensive diagnostic visualization
plotter = BBPlotter(result, data=counts)
fig = plotter.plot_diagnostics(figsize=(15, 10))
plt.show()
```

### Interactive Analysis

```python
# Interactive Plotly visualization (requires plotly)
interactive_fig = plotter.plot_interactive()
if interactive_fig:
    interactive_fig.show()
```

### Comparison Analysis

```python
from bb_plotting import plot_comparison, plot_sensitivity_analysis

# Compare results with different parameters
results = [
    bayesian_blocks_counts(data, p0=0.001),
    bayesian_blocks_counts(data, p0=0.01), 
    bayesian_blocks_counts(data, p0=0.1)
]
labels = ['Conservative', 'Moderate', 'Liberal']

fig = plot_comparison(results, labels)
plt.show()

# Parameter sensitivity analysis
p0_values = np.logspace(-4, -1, 20)
fig = plot_sensitivity_analysis(data, p0_values, 
                               lambda d, p0: bayesian_blocks_counts(d, p0=p0))
plt.show()
```

## 🔬 Advanced Features

### Bootstrap Confidence Intervals

```python
from advanced_utils import bootstrap_confidence_intervals

def algorithm(data):
    return bayesian_blocks_counts(data, p0=0.05)

# Compute confidence intervals
conf_result = bootstrap_confidence_intervals(
    data=counts,
    algorithm_func=algorithm,
    n_bootstrap=1000,
    confidence_level=0.95,
    n_jobs=4  # Parallel processing
)

print(f"Number of blocks: {len(conf_result.result.block_value)}")
print(f"95% confidence intervals computed from {len(conf_result.bootstrap_results)} bootstrap samples")
```

### Cross-Validation for Parameter Selection

```python
from advanced_utils import cross_validate_parameters

# Define parameter grid
param_grid = {
    'p0': [0.001, 0.01, 0.05, 0.1, 0.2],
    'min_block_size': [1, 2, 5]
}

# Find optimal parameters
cv_result = cross_validate_parameters(
    data=counts,
    param_grid=param_grid,
    cv_folds=5,
    scoring='log_likelihood'
)

print(f"Best parameters: p0={cv_result.best_config.p0}")
print(f"Best score: {cv_result.best_score:.2f}")

# Use best parameters
optimal_result = bayesian_blocks(counts, config=cv_result.best_config)
```

### Streaming Analysis

```python
from advanced_utils import StreamingBayesianBlocks

# Set up streaming processor
config = BBConfig(p0=0.05)
streaming = StreamingBayesianBlocks(config, buffer_size=100)

# Process data incrementally
for batch in data_batches:
    result = streaming.update(batch)
    if result is not None:
        print(f"Updated model: {len(result.block_value)} blocks")

# Get final result
final_result = streaming.finalize()
```

### One-Line Analysis

```python
from advanced_utils import quick_analysis

# Comprehensive analysis with automatic parameter selection
analysis = quick_analysis(data, show_plots=True, confidence_intervals=True)

result = analysis['result']
best_config = analysis['best_config'] 
cv_scores = analysis['cv_scores']
confidence = analysis.get('confidence')  # If requested
diagnostic_plot = analysis.get('diagnostic_plot')  # If requested
```

## 🧪 Algorithm Details

### Bayesian Blocks Method

The Bayesian Blocks algorithm finds the optimal segmentation of data by maximizing a fitness function while penalizing model complexity:

**Objective Function:**

```
F = ∑ᵢ fitness(block_i) - γ × (number_of_blocks)
```

Where:

- **fitness(block_i)**: Log-likelihood of data in block i under optimal model
- **γ (gamma)**: Penalty parameter controlling model complexity
- **p0**: Target false positive rate, used to set γ via Scargle (2013) prior

### Supported Models

Data Type     | Model           | Fitness Function            | Use Cases
------------- | --------------- | --------------------------- | ----------------------------------------
**Events**    | Poisson Process | `k*log(k/T)`                | Astronomy, neuroscience, web analytics
**Counts**    | Poisson         | `k*log(k/T)`                | Epidemiology, quality control, economics
**Bernoulli** | Binomial        | `k*log(p) + (n-k)*log(1-p)` | A/B testing, conversion analysis

### Dynamic Programming Solution

The algorithm uses O(N²) dynamic programming to find the globally optimal segmentation:

```python
# Simplified algorithm outline
for j in range(1, N+1):
    for i in range(j):
        score = opt[i] + fitness(block[i:j]) - gamma
        if score > opt[j]:
            opt[j] = score
            best_predecessor[j] = i
```

## 📊 Performance

### Computational Complexity

- **Time**: O(N²) where N is the number of data points
- **Space**: O(N) for dynamic programming tables
- **Parallel**: Bootstrap and cross-validation support multiprocessing

### Benchmarks

Typical performance on modern hardware:

- **1,000 data points**: ~10ms
- **10,000 data points**: ~1s
- **100,000 data points**: ~100s

For larger datasets, consider:

- Increasing `p0` to reduce computational cost
- Using streaming analysis for real-time processing
- Segmenting data into overlapping windows

## 🔧 Configuration Reference

### BBConfig Parameters

```python
config = BBConfig(
    p0=0.05,              # False positive rate (0 < p0 < 1)
    penalty=None,         # Direct penalty (overrides p0 if set)
    min_block_size=1,     # Minimum data points per block
    max_blocks=None,      # Maximum number of blocks (not implemented)
    method="dp"           # Algorithm method ("dp" only currently)
)
```

### Choosing Parameters

**p0 (False Positive Rate)**:

- `p0=0.001`: Very conservative, few changepoints
- `p0=0.05`: Balanced (recommended starting point)
- `p0=0.2`: Liberal, more changepoints

**min_block_size**:

- Use larger values for noisy data
- Ensures statistical significance of each block
- Trade-off between resolution and stability

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Add tests** for new functionality
4. **Run tests** (`pytest tests/`)
5. **Commit** changes (`git commit -m 'Add amazing feature'`)
6. **Push** to branch (`git push origin feature/amazing-feature`)
7. **Create** a Pull Request

### Development Setup

```bash
git clone https://github.com/yourusername/bayesian_blocks.git
cd bayesian_blocks

# Install development dependencies
pip install -e .
pip install pytest hypothesis black mypy

# Run tests
pytest tests/ -v

# Run integration tests
python tests/test_integration.py

# Format code
black bayesian_blocks/

# Type checking
mypy bayesian_blocks/
```

## 📖 References

### Primary Reference

**Scargle, J. D., Norris, J. P., Jackson, B., & Chiang, J. (2013)**<br>
"Studies in Astronomical Time Series Analysis. VI. Bayesian Block Representations"<br>
_The Astrophysical Journal_, 764(2), 167<br>
[doi:10.1088/0004-637X/764/2/167](https://doi.org/10.1088/0004-637X/764/2/167)

### Related Work

1. **Jackson, B., Scargle, J. D., Barnes, D., Arabhi, S., Alt, A., Gioumousis, P., ... & Tsai, T. T. (2005)**<br>
  "An algorithm for optimal partitioning of data on an interval"<br>
  _IEEE Signal Processing Letters_, 12(2), 105-108

2. **Scargle, J. D. (1998)**<br>
  "Studies in astronomical time series analysis. V. Bayesian blocks, a new method to analyze structure in photon counting data"<br>
  _The Astrophysical Journal_, 504(1), 405

3. **Knuth, K. H. (2006)**<br>
  "Optimal data-based binning for histograms"<br>
  _arXiv preprint physics/0605197_

## 📜 License

This project is licensed under the MIT License - see the <LICENSE> file for details.

## 🙏 Acknowledgments

- **Jeffrey Scargle** for developing the Bayesian Blocks algorithm
- **Astropy community** for inspiration and reference implementations
- **Contributors** who helped improve this library

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/diogoribeiro7/bayesian_blocks/issues)
- **Discussions**: [GitHub Discussions](https://github.com/diogoribeiro7/bayesian_blocks/discussions)
- **Email**: your.email@example.com

--------------------------------------------------------------------------------

**Happy changepoint detecting!** 🎯📈
