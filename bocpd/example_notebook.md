# BOCPD Example Notebook

This is a Jupyter notebook example showing how to use the BOCPD package. This file provides the markdown content that would be in a Jupyter notebook.

## Setup and Imports

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

# Import BOCPD components
from bocpd import BOCPD, BOCPDConfig
from bocpd import ConstantHazard, ScheduledHazard, BoostedBoundaryHazard
from bocpd_plotting import plot_run_length_heatmap, plot_cp_probability
```

## 1. Basic Example with Synthetic Data

Let's start with a simple example using synthetic binary data with a clear changepoint.

```python
# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data with a changepoint at t=100
n_before = 100
n_after = 100
x1 = np.random.binomial(1, 0.1, size=n_before)  # Low probability
x2 = np.random.binomial(1, 0.7, size=n_after)   # High probability
x = np.concatenate([x1, x2])

# Visualize the data
plt.figure(figsize=(12, 3))
plt.step(np.arange(len(x)), x, where='post')
plt.axvline(x=n_before, color='r', linestyle='--', label='True changepoint')
plt.title('Synthetic Binary Data with Changepoint')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
```

Now let's run BOCPD with a constant hazard function:

```python
# Create a model with constant hazard
hazard = ConstantHazard(mean_run_length=50.0)
config = BOCPDConfig(
    alpha0=1.0,                     # Beta prior parameter
    beta0=1.0,                      # Beta prior parameter
    max_run_length=200,             # Maximum run length to track
    store_run_length_posterior=True # Save the full posterior for visualization
)
model = BOCPD(hazard, config)

# Process the data
result = model.run(x)

# Extract results
cp_prob = result.cp_prob
map_rl = result.map_run_length
pred_mean = result.pred_mean
run_length_posterior = result.run_length_posterior

print(f"Highest CP probability at t={np.argmax(cp_prob)}")
print(f"CP probability at true changepoint (t=100): {cp_prob[100]:.4f}")
```

Let's visualize the results:

```python
# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 3, 1]})

# Plot the data
axes[0].step(np.arange(len(x)), x, where='post')
axes[0].axvline(x=n_before, color='r', linestyle='--', label='True changepoint')
axes[0].set_title('Binary Data')
axes[0].set_ylabel('Value')
axes[0].legend()

# Plot run-length posterior heatmap
im = axes[1].imshow(run_length_posterior.T, aspect='auto', origin='lower', 
                    interpolation='nearest', cmap='viridis')
axes[1].set_title('Run-Length Posterior')
axes[1].set_ylabel('Run Length')
plt.colorbar(im, ax=axes[1])

# Plot changepoint probability
axes[2].plot(np.arange(len(cp_prob)), cp_prob)
axes[2].axvline(x=n_before, color='r', linestyle='--', label='True changepoint')
axes[2].axhline(y=0.5, color='g', linestyle='--', label='Threshold = 0.5')
axes[2].set_title('Changepoint Probability')
axes[2].set_xlabel('Time')
axes[2].set_ylabel('P(CP)')
axes[2].set_ylim(0, 1)
axes[2].legend()

plt.tight_layout()
plt.show()
```

## 2. Time-of-Day Patterns with Boundary-Boosted Hazard

Let's simulate a scenario with daily patterns where changepoints are more likely at specific times of day.

```python
# Simulate daily data (96 points per day = 15-min intervals)
points_per_day = 96
days = 10

# Create pattern with different segments within each day
def generate_day_pattern(p_morning=0.1, p_midday=0.6, p_afternoon=0.3, p_night=0.05):
    # Each day has 4 segments with different probabilities
    morning = np.random.binomial(1, p_morning, size=points_per_day // 4)
    midday = np.random.binomial(1, p_midday, size=points_per_day // 4)
    afternoon = np.random.binomial(1, p_afternoon, size=points_per_day // 4)
    night = np.random.binomial(1, p_night, size=points_per_day // 4)
    return np.concatenate([morning, midday, afternoon, night])

# First 5 days have one pattern
pattern1 = np.concatenate([generate_day_pattern() for _ in range(5)])

# Next 5 days have a different pattern (higher activity)
pattern2 = np.concatenate([generate_day_pattern(p_morning=0.3, p_midday=0.8) for _ in range(5)])

# Combine patterns
x_daily = np.concatenate([pattern1, pattern2])

# The true changepoint is at day 5 boundary
true_cp = 5 * points_per_day

# Create time axis for plotting
start_time = datetime(2025, 1, 1)
time_points = [start_time + timedelta(minutes=15*i) for i in range(len(x_daily))]
```

Now let's run BOCPD with different hazard functions:

```python
# 1. Constant hazard (baseline)
hazard_constant = ConstantHazard(mean_run_length=points_per_day * 2)
model_constant = BOCPD(hazard_constant)
result_constant = model_constant.run(x_daily)

# 2. Boundary-boosted hazard (enhance detection at day boundaries)
base_hazard = ConstantHazard(mean_run_length=points_per_day * 2)
hazard_boosted = BoostedBoundaryHazard(
    base=base_hazard, 
    period=points_per_day, 
    boundary_indices=frozenset([0]),  # Boost at t % period == 0 (day boundaries)
    boost_factor=10.0
)
model_boosted = BOCPD(hazard_boosted)
result_boosted = model_boosted.run(x_daily)

# 3. Scheduled hazard (custom schedule throughout the day)
# Higher hazard at day boundaries and at segment transitions within the day
schedule = np.full(points_per_day, 0.01)  # Default low hazard
schedule[0] = 0.3                         # Day boundary (midnight)
schedule[points_per_day // 4] = 0.2       # Morning->Midday transition
schedule[points_per_day // 2] = 0.2       # Midday->Afternoon transition
schedule[3 * points_per_day // 4] = 0.2   # Afternoon->Night transition

hazard_scheduled = ScheduledHazard(schedule=schedule, period=points_per_day)
model_scheduled = BOCPD(hazard_scheduled)
result_scheduled = model_scheduled.run(x_daily)
```

Let's compare the performance of different hazard functions:

```python
# Plot results
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot the data
axes[0].step(time_points, x_daily, where='post')
axes[0].set_title('Daily Pattern Binary Data')
axes[0].set_ylabel('Value')
axes[0].axvline(x=time_points[true_cp], color='r', linestyle='--', linewidth=2, label='True Pattern Change')

# Mark day boundaries
for day in range(1, days):
    day_boundary = day * points_per_day
    if day_boundary != true_cp:  # Don't duplicate the true CP line
        axes[0].axvline(x=time_points[day_boundary], color='gray', linestyle=':', alpha=0.5)

# Format x-axis for dates
axes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M'))
axes[0].legend()

# Plot changepoint probabilities for all methods
axes[1].plot(time_points, result_constant.cp_prob, label='Constant Hazard', alpha=0.7)
axes[1].plot(time_points, result_boosted.cp_prob, label='Boundary-Boosted Hazard', alpha=0.7)
axes[1].plot(time_points, result_scheduled.cp_prob, label='Scheduled Hazard', alpha=0.7)
axes[1].axvline(x=time_points[true_cp], color='r', linestyle='--', linewidth=2, label='True Pattern Change')
axes[1].axhline(y=0.5, color='g', linestyle='--', label='Threshold = 0.5')
axes[1].set_title('Changepoint Probability Comparison')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('P(CP)')
axes[1].set_ylim(0, 1)
axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M'))
axes[1].legend()

plt.tight_layout()
plt.show()
```

## 3. Online Processing Example

Let's demonstrate how to use BOCPD in an online (streaming) scenario:

```python
# Generate some streaming data
np.random.seed(123)
stream_length = 200
p1 = 0.1  # Initial probability
p2 = 0.7  # Probability after changepoint
true_cp = 100

# Initialize model
hazard = ConstantHazard(mean_run_length=50.0)
model = BOCPD(hazard)

# Storage for online results
cp_probs = []
map_rls = []
pred_means = []
alerts = []  # Store when we detect a changepoint

# Process data one point at a time
for t in range(stream_length):
    # Generate next observation
    if t < true_cp:
        x_t = np.random.binomial(1, p1)
    else:
        x_t = np.random.binomial(1, p2)
    
    # Process point
    result = model.update(x_t)
    
    # Store results
    cp_probs.append(result["cp_prob"])
    map_rls.append(result["map_run_length"])
    pred_means.append(result["pred_mean"])
    
    # Alert if changepoint probability exceeds threshold
    if result["cp_prob"] > 0.5:
        alerts.append(t)
        print(f"🚨 Changepoint detected at t={t} with probability {result['cp_prob']:.4f}")
```

Visualize the online processing results:

```python
# Plot results
fig, axes = plt.subplots(3, 1, figsize=(12, 9))

# Generate the full sequence for plotting
x_stream = np.concatenate([
    np.random.binomial(1, p1, size=true_cp),
    np.random.binomial(1, p2, size=stream_length - true_cp)
])

# Plot data
axes[0].step(np.arange(stream_length), x_stream, where='post')
axes[0].set_title('Binary Data Stream')
axes[0].set_ylabel('Value')
axes[0].axvline(x=true_cp, color='r', linestyle='--', label='True changepoint')
for alert in alerts:
    axes[0].axvline(x=alert, color='g', linestyle=':', alpha=0.7)
axes[0].legend()

# Plot changepoint probability
axes[1].plot(np.arange(stream_length), cp_probs)
axes[1].set_title('Online Changepoint Probability')
axes[1].set_ylabel('P(CP)')
axes[1].axhline(y=0.5, color='g', linestyle='--', label='Alert threshold')
axes[1].axvline(x=true_cp, color='r', linestyle='--', label='True changepoint')
axes[1].set_ylim(0, 1)
axes[1].legend()

# Plot run length
axes[2].plot(np.arange(stream_length), map_rls)
axes[2].set_title('Maximum A Posteriori Run Length')
axes[2].set_xlabel('Time')
axes[2].set_ylabel('Run Length')
axes[2].axvline(x=true_cp, color='r', linestyle='--', label='True changepoint')
axes[2].legend()

plt.tight_layout()
plt.show()
```

## 4. Working with Real Data from CSV

Let's demonstrate how to use the data loading utilities with a synthetic CSV file:

```python
# First, create a synthetic CSV file with timestamped events
csv_filename = "simulated_events.csv"

# Generate synthetic timestamps with a pattern change
start_date = datetime(2025, 1, 1)
timestamps = []
values = []

# First week: sparse activity
for day in range(7):
    # Generate 5-10 events per day with low values
    for _ in range(np.random.randint(5, 11)):
        hour = np.random.randint(0, 24)
        minute = np.random.randint(0, 60)
        timestamps.append(start_date + timedelta(days=day, hours=hour, minutes=minute))
        values.append(np.random.uniform(0.2, 0.6))  # Lower values

# Second week: increased activity
for day in range(7, 14):
    # Generate 15-25 events per day with higher values
    for _ in range(np.random.randint(15, 26)):
        hour = np.random.randint(0, 24)
        minute = np.random.randint(0, 60)
        timestamps.append(start_date + timedelta(days=day, hours=hour, minutes=minute))
        values.append(np.random.uniform(0.5, 0.9))  # Higher values

# Create DataFrame and sort by timestamp
df = pd.DataFrame({
    "timestamp": timestamps,
    "value": values
})
df = df.sort_values("timestamp")

# Save to CSV
df.to_csv(csv_filename, index=False)
print(f"Created {len(df)} events spanning {(timestamps[-1] - timestamps[0]).days + 1} days")
```

Now load the data and analyze it with BOCPD:

```python
from data_loader import load_binary_from_csv

# Load data using data_loader
bin_minutes = 60  # 1-hour bins
binary_data, bins_per_day, time_bins = load_binary_from_csv(
    csv_filename,
    timestamp_col="timestamp",
    value_col="value",
    value_threshold=0.5,  # Only count high-value events
    bin_minutes=bin_minutes,
    start_hour=0,
    return_time_bins=True
)

print(f"Loaded data with {bins_per_day} bins per day, {len(binary_data)} total bins")
print(f"Number of active bins: {binary_data.sum()}")

# Create model with boundary-boosted hazard (weekly pattern)
base_hazard = ConstantHazard(mean_run_length=bins_per_day * 3)  # 3-day mean run length
hazard = BoostedBoundaryHazard(
    base=base_hazard,
    period=bins_per_day * 7,  # Weekly period
    boundary_indices=frozenset([0]),  # Boost at weekly boundaries
    boost_factor=8.0
)

model = BOCPD(hazard)
result = model.run(binary_data)

# Plot results
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot binary data
axes[0].step(time_bins[:-1], binary_data, where='post')
axes[0].set_title('Binned Event Data (1-hour bins)')
axes[0].set_ylabel('Active')
axes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M'))

# Expected weekly boundary
weekly_boundary = time_bins[bins_per_day * 7]
axes[0].axvline(x=weekly_boundary, color='r', linestyle='--', label='Weekly Boundary')

# Plot changepoint probability
axes[1].plot(time_bins[:-1], result.cp_prob)
axes[1].axvline(x=weekly_boundary, color='r', linestyle='--', label='Weekly Boundary')
axes[1].axhline(y=0.5, color='g', linestyle='--', label='Threshold = 0.5')
axes[1].set_title('Changepoint Probability')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('P(CP)')
axes[1].set_ylim(0, 1)
axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M'))
axes[1].legend()

plt.tight_layout()
plt.show()
```

## 5. Hyperparameter Selection

Let's examine how prior parameters affect detection sensitivity:

```python
# Generate data with a subtle changepoint
np.random.seed(42)
n_before = 100
n_after = 100
# Subtle change: 0.2 -> 0.3
x1 = np.random.binomial(1, 0.2, size=n_before)
x2 = np.random.binomial(1, 0.3, size=n_after)
x_subtle = np.concatenate([x1, x2])

# Try different prior parameters
prior_configs = [
    {"alpha0": 0.5, "beta0": 0.5, "name": "Jeffreys-like (α₀=β₀=0.5)"},
    {"alpha0": 1.0, "beta0": 1.0, "name": "Uniform (α₀=β₀=1.0)"},
    {"alpha0": 5.0, "beta0": 5.0, "name": "Strong prior (α₀=β₀=5.0)"},
    {"alpha0": 0.2, "beta0": 0.8, "name": "Informative (p≈0.2)"}
]

# Run BOCPD with each configuration
results = []
for config in prior_configs:
    model = BOCPD(
        ConstantHazard(mean_run_length=50.0),
        BOCPDConfig(alpha0=config["alpha0"], beta0=config["beta0"])
    )
    result = model.run(x_subtle)
    results.append({"name": config["name"], "result": result})

# Plot comparison
plt.figure(figsize=(12, 6))

# Plot data
plt.subplot(2, 1, 1)
plt.step(np.arange(len(x_subtle)), x_subtle, where='post')
plt.axvline(x=n_before, color='r', linestyle='--', label='True changepoint')
plt.title('Binary Data with Subtle Changepoint')
plt.ylabel('Value')
plt.legend()

# Plot CP probabilities
plt.subplot(2, 1, 2)
for res in results:
    plt.plot(np.arange(len(x_subtle)), res["result"].cp_prob, label=res["name"], alpha=0.7)
plt.axvline(x=n_before, color='r', linestyle='--', label='True changepoint')
plt.axhline(y=0.5, color='g', linestyle='--', label='Threshold = 0.5')
plt.title('Changepoint Probability with Different Priors')
plt.xlabel('Time')
plt.ylabel('P(CP)')
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.show()
```

## 6. Analyzing Performance Metrics

Let's evaluate the detection performance with different hazard functions:

```python
# Generate a longer sequence with multiple changepoints
np.random.seed(789)
seq_length = 1000
true_cps = [200, 500, 800]

# Generate data with alternating probabilities
x_multi = np.zeros(seq_length, dtype=bool)
p = 0.1
for i in range(seq_length):
    if i in true_cps:
        p = 0.7 if p == 0.1 else 0.1
    x_multi[i] = np.random.binomial(1, p)

# Test different hazard configurations
hazard_configs = [
    {"hazard": ConstantHazard(mean_run_length=100), "name": "Constant (λ=100)"},
    {"hazard": ConstantHazard(mean_run_length=200), "name": "Constant (λ=200)"},
    {"hazard": BoostedBoundaryHazard(
        base=ConstantHazard(mean_run_length=150),
        period=100,
        boundary_indices=frozenset([0]),
        boost_factor=5.0
     ), "name": "Boosted (period=100)"}
]

# Run BOCPD with each hazard
results_multi = []
for config in hazard_configs:
    model = BOCPD(config["hazard"])
    result = model.run(x_multi)
    results_multi.append({"name": config["name"], "result": result})

# Function to calculate detection metrics
def calculate_metrics(cp_probs, true_cps, threshold=0.5, window=10):
    """
    Calculate detection performance metrics.
    
    Parameters:
    -----------
    cp_probs : array
        Changepoint probabilities
    true_cps : list
        True changepoint positions
    threshold : float
        Detection threshold
    window : int
        Window size for matching detections to true changepoints
        
    Returns:
    --------
    dict
        Dictionary with performance metrics
    """
    # Find detected changepoints
    detected = np.where(cp_probs >= threshold)[0].tolist()
    
    # Count true positives
    tp = 0
    matched_true = set()
    for d in detected:
        for t in true_cps:
            if abs(d - t) <= window and t not in matched_true:
                tp += 1
                matched_true.add(t)
                break
    
    # Calculate metrics
    precision = tp / max(1, len(detected))
    recall = tp / len(true_cps)
    f1 = 2 * precision * recall / max(1e-10, precision + recall)
    
    # Average detection delay for true positives
    delays = []
    for t in true_cps:
        matched_detections = [d for d in detected if abs(d - t) <= window]
        if matched_detections:
            delay = min(d - t for d in matched_detections if d >= t)
            delays.append(delay if delay >= 0 else 0)
    
    avg_delay = np.mean(delays) if delays else np.nan
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detected": detected,
        "true_positives": tp,
        "false_positives": len(detected) - tp,
        "false_negatives": len(true_cps) - tp,
        "avg_delay": avg_delay
    }

# Calculate metrics for each model
metrics = []
for res in results_multi:
    m = calculate_metrics(res["result"].cp_prob, true_cps)
    metrics.append({**m, "name": res["name"]})

# Print metrics
print("Performance Metrics:")
for m in metrics:
    print(f"\n{m['name']}:")
    print(f"  Precision: {m['precision']:.2f}")
    print(f"  Recall: {m['recall']:.2f}")
    print(f"  F1 Score: {m['f1']:.2f}")
    print(f"  Average Delay: {m['avg_delay']:.1f}")
    print(f"  True Positives: {m['true_positives']}")
    print(f"  False Positives: {m['false_positives']}")
    print(f"  False Negatives: {m['false_negatives']}")

# Plot results
plt.figure(figsize=(14, 10))

# Plot data
plt.subplot(2, 1, 1)
plt.step(np.arange(seq_length), x_multi, where='post')
for cp in true_cps:
    plt.axvline(x=cp, color='r', linestyle='--')
plt.title('Binary Data with Multiple Changepoints')
plt.ylabel('Value')

# Plot CP probabilities

