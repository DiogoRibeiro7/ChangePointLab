# BOCPD: Bayesian Online Changepoint Detection for Binary Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python implementation of Bayesian Online Changepoint Detection (BOCPD) specifically designed for binary (Bernoulli) data, with extensions for other distributions. This package implements the algorithm from Adams and MacKay (2007) with a flexible modular design that supports multiple likelihood models and hazard functions.

## Features

- **Online processing**: Process data points one at a time with constant memory usage
- **Pluggable likelihood models**:
  - Beta-Bernoulli for binary data (fully implemented)
  - Poisson-Gamma for count data (skeleton provided)
  - Gaussian-NIW for continuous data (skeleton provided)
- **Flexible hazard functions**:
  - Constant hazard (classic BOCPD)
  - Time-dependent scheduled hazard (e.g., time-of-day patterns)
  - Boundary-boosted hazard (enhance detection at specific points)
- **Numerically stable**: Robust handling of underflow and long sequences
- **Full posterior access**: Access the complete run-length posterior distribution
- **DST-safe binning**: Proper handling of timezone transitions for real-world data
- **Command-line interface**: Process data from CSV files using the CLI
- **Visualization tools**: Plot run-length posterior heatmaps and changepoint probabilities

## Installation

Clone the repository and ensure you have the required dependencies:

```bash
git clone https://github.com/yourusername/bocpd.git
cd bocpd
pip install -r requirements.txt

# Alternatively, install the package directly
pip install .

# For development
pip install -e ".[dev]"
```

Required dependencies: NumPy, Matplotlib, Pandas

## Quick Start

```python
import numpy as np
from bocpd import BOCPD, ConstantHazard

# Create synthetic data with a changepoint
x1 = np.random.binomial(1, 0.1, size=50)  # Low probability
x2 = np.random.binomial(1, 0.8, size=50)  # High probability
x = np.concatenate([x1, x2])

# Create model with constant hazard
hazard = ConstantHazard(mean_run_length=50.0)
model = BOCPD(hazard)

# Process the data
result = model.run(x)

# Access results
print(f"CP probability at t=50: {result.cp_prob[50]:.4f}")
print(f"MAP run length at t=60: {result.map_run_length[60]}")
print(f"Predictive mean at t=30: {result.pred_mean[30]:.4f}")
```

## Usage Scenarios

### DST-safe Binning

Use `--timezone Europe/Lisbon` (or your IANA zone) for proper handling of timezone transitions:
- Naive timestamps are tz-localized with `nonexistent='shift_forward'` and `ambiguous='NaT'` (dropped)
- Bin edges are aligned to local "wall clock" days via `start_hour`
- Left-closed intervals with `include_lowest=True` for reliable binning

### 1. Event Stream Monitoring

Detect changes in binary event streams, such as:
- User behavior changes (clicks, conversions)
- System state transitions (online/offline)
- Anomaly detection in IoT sensor data (active/inactive)

### 2. Time-of-Day Analysis

Use scheduled hazards to detect changes in diurnal patterns:
- Daily routine changes
- Shift changes in industrial settings
- Website traffic pattern shifts

### 3. Online Processing

Process data as it arrives for real-time detection:
- Network traffic monitoring
- User activity tracking
- Process control systems

### 4. Offline Analysis

Analyze historical data for pattern discovery:
- Post-hoc detection of behavioral changes
- Identification of seasonal boundaries
- Research on historical binary time series

## Command-Line Interface

The package includes a CLI for processing CSV files containing timestamped events:

```bash
python -m bocpd_cli --csv events.csv --bin-minutes 15 --mean-rl 96 --cp-threshold 0.6
```

Key parameters:
- `--csv`: Path to CSV file with timestamped events
- `--bin-minutes`: Minutes per bin for aggregating events
- `--mean-rl`: Mean run length for constant hazard
- `--schedule`: Optional comma-separated hazard values for scheduled hazard
- `--boost-boundary`: Optional boundary indices to boost hazard
- `--cp-threshold`: Threshold for flagging changepoints
- `--timezone`: IANA timezone for DST-safe binning (e.g., "Europe/Lisbon")

Or try the built-in demo:

```bash
python -m bocpd_cli --demo --days 14 --period 96
```

## Theory

### Likelihood Models

The BOCPD algorithm supports different likelihood models through a pluggable architecture:

#### Beta-Bernoulli Model (Binary Data)
- Prior distribution: Beta(α₀, β₀)
- Likelihood: Bernoulli(p)
- Posterior: Beta(α₀ + Σx, β₀ + Σ(1-x))
- Predictive distribution: Bernoulli(α/(α+β))

#### Parameter Priors (α₀, β₀)
- **Uniform:** α₀=β₀=1 (default). Balanced shrinkage toward p=0.5.
- **Jeffreys-like:** α₀=β₀=0.5. Less shrinkage in the tails; slightly sharper change detection.
- **Informative:** set α₀,β₀ so that α₀/(α₀+β₀)≈ baseline event rate. Example: baseline 10% → α₀=1, β₀=9.

### Hazard Functions

The hazard function H(r, t) controls the prior probability of a changepoint:
- H(r, t) = P(changepoint at t | run_length = r, t)
- Classic BOCPD uses constant hazard: H(r, t) = 1/λ

#### Hazard Scaling (Boundary Boost)
`BoostedBoundaryHazard` multiplies a base hazard by `boost_factor` on chosen boundaries, then clips to < 1 to stay probabilistic. If `boost_factor` is too large, values saturate at 1 and you lose contrast across run-lengths. Pick a `boost_factor` that keeps boosted hazards well below 1.

### Run-Length Distribution

The algorithm maintains a distribution over the current run length:
- r_t = 0: A changepoint just occurred
- r_t > 0: Time since the last changepoint

## API Reference

### Core Classes

#### `BOCPD`

Main class for Bayesian Online Changepoint Detection:

```python
model = BOCPD(hazard, cfg=BOCPDConfig())
```

Methods:
- `update(x_t)`: Process a single observation
- `run(x)`: Process a sequence of observations
- `reset()`: Reset the model to initial state

#### `BOCPDConfig`

Configuration for the BOCPD algorithm:

```python
cfg = BOCPDConfig(
    alpha0=1.0,             # Beta prior α parameter
    beta0=1.0,              # Beta prior β parameter
    max_run_length=512,     # Truncation for run length support
    store_run_length_posterior=True,  # Whether to store full posterior
    prune_epsilon=1e-6,     # Threshold for pruning tiny probabilities
    prune_relative=True,    # Use relative pruning (vs absolute)
    stabilizer=1e-300,      # Small floor to detect underflow
    top_k=None,             # Optional: keep only top K states each step
)
```

#### `BOCPDResult`

Results from processing a sequence:

```python
result = model.run(x)
```

Attributes:
- `cp_prob`: Changepoint probability at each time
- `map_run_length`: Maximum a posteriori run length
- `pred_mean`: One-step-ahead predictive mean
- `run_length_posterior`: Full run-length posterior distribution

### Hazard Functions

#### `ConstantHazard`

```python
hazard = ConstantHazard(mean_run_length=100.0)
```

- `mean_run_length`: Expected segment length

#### `ScheduledHazard`

```python
# Higher hazard at t % 24 == 0 (daily boundary)
schedule = [0.1 if i == 0 else 0.01 for i in range(24)]
hazard = ScheduledHazard(schedule=schedule, period=24)
```

- `schedule`: Hazard values for indices 0...period-1
- `period`: Period for cycling through the schedule

#### `BoostedBoundaryHazard`

```python
# Boost hazard at day boundaries (t % 96 == 0)
base = ConstantHazard(mean_run_length=200.0)
hazard = BoostedBoundaryHazard(
    base=base,
    period=96,
    boundary_indices=frozenset([0]),
    boost_factor=10.0
)
```

- `base`: Base hazard function
- `period`: Period for boundary check
- `boundary_indices`: Set of indices to boost
- `boost_factor`: Multiplier for the hazard at boundaries

### Likelihood Models

The package provides a flexible likelihood framework in `likelihoods.py`:

```python
from bocpd.likelihoods import BetaBernoulli, PoissonGamma

# Create a custom BOCPD model with Beta-Bernoulli likelihood
from bocpd import BOCPD, BOCPDConfig, ConstantHazard
from bocpd.likelihoods import BetaBernoulli

hazard = ConstantHazard(mean_run_length=100.0)
cfg = BOCPDConfig(max_run_length=512)
likelihood = BetaBernoulli(alpha0=0.5, beta0=0.5)  # Jeffreys-like prior
model = BOCPD(hazard=hazard, cfg=cfg, likelihood=likelihood)
```

Currently implemented models:
- `BetaBernoulli`: For binary (0/1) data (fully implemented)
- `PoissonGamma`: For count data (skeleton provided)
- `GaussianNIW`: For continuous data with Normal-Inverse-Wishart prior (skeleton provided)

## Data Loading

The package includes utilities for loading and preprocessing data:

```python
from bocpd.data_loader import load_binary_from_csv

# Load binary data from a CSV file with timestamps
binary_data, bins_per_day, time_bins = load_binary_from_csv(
    "events.csv",
    timestamp_col="event_time",
    value_col="metric",
    value_threshold=0.5,
    bin_minutes=15,
    start_hour=0,
    timezone="Europe/London",
    return_time_bins=True
)
```

## Examples

See the `example_usage.py` file for detailed examples, including:
1. Basic usage with synthetic data
2. Online processing (streaming data)
3. Custom hazard functions
4. Working with CSV data

## Performance Considerations

- Memory usage scales with `max_run_length` (truncation parameter)
- Computation time is O(T × max_run_length) for a sequence of length T
- For very long sequences, consider processing in chunks or streaming
- Use `top_k` parameter for sparse approximate filtering (retains only the top K run-length states)
- The `prune_epsilon` parameter can be adjusted to balance speed and accuracy

## References

- Adams, R. P., & MacKay, D. J. (2007). Bayesian online changepoint detection. arXiv preprint arXiv:0710.3742.
- Fearnhead, P., & Liu, Z. (2007). On-line inference for multiple changepoint problems. Journal of the Royal Statistical Society: Series B, 69(4), 589-605.

## License

MIT License (c) 2025 Diogo Ribeiro. See LICENSE file for details.
