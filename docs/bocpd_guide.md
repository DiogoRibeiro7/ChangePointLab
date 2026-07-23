# BOCPD Parameter Selection Guide

This guide provides detailed information on selecting appropriate parameters for the Bayesian Online Changepoint Detection (BOCPD) algorithm implemented in this package. Proper parameter selection is crucial for effective changepoint detection.

## Prior Parameters (α₀, β₀)

The Beta prior parameters control the initial beliefs about the Bernoulli probability parameter:

### α₀, β₀ (alpha0, beta0)

- **Default**: α₀ = β₀ = 1.0 (uniform prior)
- **Interpretation**:

  - α₀ = β₀ = 1: Uniform prior (no preference)
  - α₀ > β₀: Prior belief favors p > 0.5
  - α₀ < β₀: Prior belief favors p < 0.5
  - Large values (e.g., α₀ = β₀ = 10): Strong prior, requires more evidence to change
  - Small values (e.g., α₀ = β₀ = 0.5): Weak prior, adapts quickly to data

- **Recommendation**:

  - For exploratory analysis, start with α₀ = β₀ = 1.0
  - If you have domain knowledge about the baseline probability, set α₀/(α₀+β₀) to that value
  - The sum α₀+β₀ controls how quickly the model adapts to new data; smaller values adapt faster

- **Examples**:

  - Binary sensor with 5% baseline activation: α₀=0.5, β₀=9.5
  - Click-through rate around 1%: α₀=0.1, β₀=9.9
  - Balanced class with no prior knowledge: α₀=1.0, β₀=1.0

## Hazard Function Parameters

> **Boundary boosts & saturation.** The boosted hazard is clipped to < 1. Overly large `boost_factor` values can saturate the hazard and blur differences among run-length states. Prefer moderate boosts to preserve contrast.

### Mean Run Length (for ConstantHazard)

- **Default**: No default, must be specified
- **Interpretation**: Expected segment length in time steps
- **Calculation**: hazard = 1/mean_run_length
- **Recommendation**:

  - Set to the expected duration between changepoints
  - For daily data with weekly patterns: mean_run_length ≈ 7 days
  - For no prior expectation: set to half the sequence length

- **Examples**:

  - 15-minute bins with expected changes every 3 days: mean_run_length = 3 _24_ 4 = 288
  - Hourly data with expected changes every week: mean_run_length = 7 * 24 = 168

### Schedule and Period (for ScheduledHazard)

- **Period**: Number of time steps in one complete cycle
- **Schedule**: List of hazard values for each position in the cycle
- **Recommendation**:

  - Set period to match natural cycles in your data (e.g., 24 for hourly data with daily patterns)
  - Use higher hazard values at times when changes are more likely
  - Keep most values low (0.01-0.05) and increase at expected boundaries (0.1-0.5)

- **Examples**:

  - Daily boundary detection (hourly data): 

    ```python
    period = 24
    schedule = [0.2 if i == 0 else 0.01 for i in range(period)]
    ```

  - Business hours transition (15-min bins):

    ```python
    period = 96  # 24 hours * 4 bins per hour
    schedule = [0.1 if i in [32, 68] else 0.01 for i in range(period)]  # 8am and 5pm
    ```

### Boundary Indices and Boost Factor (for BoostedBoundaryHazard)

- **Boundary Indices**: Set of indices within the period to apply boosting
- **Boost Factor**: Multiplier for the base hazard at boundary points
- **Recommendation**:

  - Use this when you have specific points in a cycle where changes are more likely
  - Set boundary_indices to these specific points (as indices within the period)
  - Boost factor of 5-10 is typically sufficient; higher values force detection

- **Examples**:

  - Weekly pattern with higher weekend/weekday transitions:

    ```python
    # Daily data with weekly pattern
    base = ConstantHazard(mean_run_length=30)
    hazard = BoostedBoundaryHazard(
        base=base,
        period=7,
        boundary_indices=frozenset([0, 5]),  # Sunday and Friday
        boost_factor=8.0
    )
    ```

## Algorithm Parameters

### Maximum Run Length (max_run_length)

- **Default**: 512
- **Interpretation**: Maximum possible run length to consider
- **Recommendation**:

  - Set to at least 2-3 times the expected longest segment
  - Trade-off: larger values increase memory usage and computation time
  - Too small values can miss long-duration segments

- **Examples**:

  - For hourly data with expected changes every 1-2 weeks: max_run_length = 3 _14_ 24 = 1008
  - For 5-minute data with expected changes every 1-2 days: max_run_length = 3 _2_ 24 * 12 = 1728

## Practical Parameter Selection

## Posterior Probabilities and Alert Extraction

`BOCPD.run(...)` updates the online run-length posterior. With the default
configuration, `cp_prob[t]` is the canonical unscaled posterior probability
`P(r_t = 0 | x_1:t)` under the configured hazard and Beta-Bernoulli predictive
model.

Changepoint alerts are extracted after inference. Use `BOCPDAlertConfig` to set
an explicit `probability_threshold`, optional local-peak filtering, optional
run-length-reset filtering, and `min_spacing` cooldown. This policy affects only
wrapper/CLI alert indices; it does not change `cp_prob`, `map_run_length`, or the
stored run-length posterior.

`cp_scale` is deprecated. Values other than `1.0` preserve legacy boosted
behavior for comparison runs, but the resulting `cp_prob` is a compatibility
score and diagnostics mark it as not calibrated.

### Time-Scale Considerations

When working with real-world data, consider the natural time scales:

1. **Bin Size**: How to aggregate events into binary indicators

  - Too fine: Sparse data with many zeros
  - Too coarse: Loss of temporal detail

2. **Expected Change Frequency**: How often changepoints typically occur

  - Set mean_run_length based on this expectation
  - For seasonal data, consider scheduled or boosted hazards

3. **Memory Requirements**: Longer max_run_length requires more memory

  - For very long sequences, consider chunking or streaming

### Example Scenarios

#### 1\. Daily Website Traffic Patterns (hourly data)

```python
# Configuration for detecting changes in daily website traffic patterns
config = BOCPDConfig(
    alpha0=1.0,
    beta0=1.0,
    max_run_length=24*7*2  # 2 weeks worth of hours
)

# Use boosted hazard at day boundaries (midnight)
hazard = BoostedBoundaryHazard(
    base=ConstantHazard(mean_run_length=24*3),  # 3 days average
    period=24,  # 24 hours
    boundary_indices=frozenset([0]),  # Midnight
    boost_factor=5.0
)
```

#### 2\. IoT Sensor Activity (5-minute bins)

```python
# Configuration for IoT sensor binary activation patterns
config = BOCPDConfig(
    alpha0=0.2,  # Favor inactive state
    beta0=1.8,   # (p=0.1 prior expectation)
    max_run_length=12*24*3  # 3 days worth of 5-min bins
)

# Use scheduled hazard with time-of-day pattern
bins_per_day = 12*24  # 288 bins per day (5-min intervals)
schedule = np.ones(bins_per_day) * 0.01  # Base hazard

# Higher hazard at key times (8am, 12pm, 5pm, 10pm)
key_times = [
    8*12,    # 8am
    12*12,   # 12pm
    17*12,   # 5pm
    22*12    # 10pm
]
for t in key_times:
    schedule[t] = 0.1  # Higher hazard at these times

hazard = ScheduledHazard(schedule=schedule, period=bins_per_day)
```

#### 3\. User Interaction Events (irregularly spaced)

```python
# First bin events into 15-minute intervals
binary_data, bins_per_day = load_binary_from_csv(
    "user_events.csv",
    timestamp_col="event_time",
    bin_minutes=15
)

# Configuration for detecting user behavior changes
config = BOCPDConfig(
    alpha0=0.5,
    beta0=4.5,  # Prior expectation ~10% event rate
    max_run_length=bins_per_day*5  # 5 days worth
)

# Simple constant hazard for baseline
hazard = ConstantHazard(mean_run_length=bins_per_day)  # ~1 day segments
```

## Evaluating Parameter Choices

To assess whether your parameter choices are appropriate:

1. **Visual Inspection**:

  - Plot the binary data and detected changepoints
  - Check if detected changepoints align with visible pattern changes
  - Examine the run-length posterior heatmap for coherent structures

2. **Sensitivity Analysis**:

  - Vary the mean_run_length parameter and observe detection stability
  - Try different prior parameters to see their effect on detection

3. **Domain Validation**:

  - Verify detected changepoints against known events or external data
  - Check if the segmentation makes sense in your application context

## Common Issues and Solutions

Issue                                   | Possible Cause                  | Solution
--------------------------------------- | ------------------------------- | ---------------------------------
Too many false positives                | mean_run_length too small       | Increase mean_run_length
Missing obvious changepoints            | mean_run_length too large       | Decrease mean_run_length
Delayed detection                       | Strong prior (large α₀+β₀)      | Reduce prior strength
Inconsistent boundary detection         | Natural cycle not accounted for | Use scheduled or boosted hazard
Slow processing                         | max_run_length too large        | Reduce max_run_length if possible
Model ignores small probability changes | Prior favors certain values     | Adjust α₀ and β₀ to be smaller

## Parameter Tuning Workflow

1. **Start with defaults**:

  - α₀ = β₀ = 1.0 (uniform prior)
  - mean_run_length = length(data) / 10
  - max_run_length = 3 * mean_run_length

2. **Adjust hazard function**:

  - If data has natural cycles, switch to scheduled/boosted hazard
  - Tune mean_run_length based on expected segment length

3. **Fine-tune priors**:

  - If you have domain knowledge about baseline probability, adjust α₀ and β₀
  - If detection is too sensitive/insensitive, adjust the sum α₀+β₀

4. **Validate and iterate**:

  - Check results against ground truth or domain knowledge
  - Adjust parameters iteratively based on performance
