# Getting Started with ChangePointLab

This tutorial introduces the fundamentals of changepoint detection and walks through basic examples using ChangePointLab.

## What are Changepoints?

Changepoints are moments in a time series where the statistical properties of the data change abruptly. These can manifest as:

- Shifts in the mean level
- Changes in variance or volatility
- Alterations in trend or seasonality
- Switches in the underlying distribution

Detecting these changes helps segment time series into homogeneous regions, identify anomalies, and understand when important transitions occur in your data.

## Installation

```bash
git clone https://github.com/DiogoRibeiro7/ChangePointLab
cd ChangePointLab
poetry install
```

## A Simple Example: COVID-19 Impact on Stock Prices

Let's examine how COVID-19 affected the stock market by detecting changepoints in the S&P 500 index.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from changepoint_lab import BOCPD, PELT, EDivisive
from changepoint_lab.algorithms.optimization.cost_functions import (
    NormalMeanVarUnknown,
    bic_penalty,
)
from changepoint_lab.algorithms.bayesian.bocpd import ConstantHazard, BOCPDConfig

# Load S&P 500 data (2019-2020)
# In a real application, you would fetch this with yfinance or a similar library
dates = pd.date_range(start='2019-01-01', end='2020-12-31', freq='B')
# Simulated data representing S&P 500 during this period
prices = np.concatenate([
    np.linspace(2500, 3300, 250) + np.random.normal(0, 30, 250),  # 2019 bull market
    np.linspace(3300, 2300, 60) + np.random.normal(0, 100, 60),   # COVID-19 crash
    np.linspace(2300, 3700, 240) + np.random.normal(0, 50, 240)   # Recovery
])

# Convert to returns (day-to-day percentage changes)
returns = 100 * np.diff(prices) / prices[:-1]

# Plot the price series
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(dates[:len(prices)], prices)
plt.title('S&P 500 Index (2019-2020)')
plt.ylabel('Price')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(dates[1:len(returns)+1], returns)
plt.title('Daily Returns (%)')
plt.ylabel('Return (%)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Method 1: PELT for Offline Detection

Let's use PELT (Pruned Exact Linear Time) to detect changes in the mean and variance of returns:

```python
# Apply PELT to detect changes in return distribution
cost_fn = NormalMeanVarUnknown()
cost_fn.precompute(returns)

# Apply PELT with BIC penalty
n = len(returns)
penalty = bic_penalty(params_per_segment=2, n=n)  # 2 parameters: mean and variance
model_pelt = PELT(cost_fn=cost_fn, penalty=penalty, min_seg_len=10)
result_pelt = model_pelt.fit_predict(returns)

# Print detected changepoints
print(f"PELT detected {len(result_pelt.indices)} changepoints:")
for cp in result_pelt.indices:
    print(f"  - {dates[cp].strftime('%Y-%m-%d')}")

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(dates[1:len(returns)+1], returns)
plt.title('Changepoints in S&P 500 Returns (PELT)')
plt.ylabel('Return (%)')

# Add vertical lines for changepoints
for cp in result_pelt.indices:
    plt.axvline(x=dates[cp], color='r', linestyle='--', alpha=0.7)
    plt.text(dates[cp], plt.ylim()[1]*0.9, dates[cp].strftime('%Y-%m-%d'),
             rotation=90, verticalalignment='top')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Method 2: BOCPD for Sequential Detection

Now let's try Bayesian Online Changepoint Detection on a binary indicator stream.
The default BOCPD wrapper uses the implemented Beta-Bernoulli likelihood path;
scalar count streams can instead pass `PoissonGamma`.

```python
negative_return = (returns < 0).astype(int)

# Initialize BOCPD detector
detector = BOCPD(
    hazard=ConstantHazard(mean_run_length=180),  # Expected segment length ~180 days
    cfg=BOCPDConfig(max_run_length=500),
)

# Process binary returns sequentially
result_bocpd = detector.fit_predict(negative_return)

# Extract changepoint probabilities
cp_probs = result_bocpd.metadata["cp_prob"]

# Visualize results
plt.figure(figsize=(12, 8))

# Plot returns
plt.subplot(2, 1, 1)
plt.plot(dates[1:len(returns)+1], returns)
plt.title('S&P 500 Returns')
plt.ylabel('Return (%)')
plt.grid(True, alpha=0.3)

# Plot changepoint probability
plt.subplot(2, 1, 2)
plt.plot(dates[1:len(returns)+1], cp_probs)
plt.title('Changepoint Probability (BOCPD on Negative-Return Indicator)')
plt.ylabel('Probability')
plt.xlabel('Date')
plt.grid(True, alpha=0.3)

# Mark high-probability changepoints
threshold = 0.4
cp_indices = np.where(cp_probs > threshold)[0]
for idx in cp_indices:
    plt.axvline(x=dates[idx+1], color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Print detected changepoints
print(f"BOCPD detected changes with probability > {threshold} on:")
for idx in cp_indices:
    print(f"  - {dates[idx+1].strftime('%Y-%m-%d')} (p={cp_probs[idx]:.2f})")
```

### Method 3: E-Divisive for Non-parametric Detection

E-Divisive doesn't make assumptions about the data distribution:

```python
# Apply E-Divisive to returns
model_ediv = EDivisive(alpha=1.0, min_size=20, R=99)
result_ediv = model_ediv.fit_predict(returns.reshape(-1, 1))

# Print detected changepoints
print(f"E-Divisive detected {len(result_ediv.indices)} changepoints:")
for cp in result_ediv.indices:
    print(f"  - {dates[cp+1].strftime('%Y-%m-%d')}")

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(dates[1:len(returns)+1], returns)
plt.title('Changepoints in S&P 500 Returns (E-Divisive)')
plt.ylabel('Return (%)')

# Add vertical lines for changepoints
for cp in result_ediv.indices:
    plt.axvline(x=dates[cp+1], color='g', linestyle='--', alpha=0.7)
    plt.text(dates[cp+1], plt.ylim()[1]*0.9, dates[cp+1].strftime('%Y-%m-%d'),
             rotation=90, verticalalignment='top')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Comparing the Methods

Let's compare the results from all three methods:

```python
# Convert changepoints to datetime for comparison
pelt_dates = [dates[cp] for cp in result_pelt.indices]
bocpd_dates = [dates[idx+1] for idx in cp_indices]
ediv_dates = [dates[cp+1] for cp in result_ediv.indices]

# Create a combined visualization
plt.figure(figsize=(12, 8))
plt.plot(dates[1:len(returns)+1], returns, alpha=0.7)
plt.title('Changepoint Detection Comparison - S&P 500 Returns (2019-2020)')
plt.ylabel('Return (%)')

# Plot changepoints from each method
for date in pelt_dates:
    plt.axvline(x=date, color='r', linestyle='--', alpha=0.7, label='PELT' if date==pelt_dates[0] else '')
    
for date in bocpd_dates:
    plt.axvline(x=date, color='b', linestyle='-.', alpha=0.7, label='BOCPD' if date==bocpd_dates[0] else '')
    
for date in ediv_dates:
    plt.axvline(x=date, color='g', linestyle=':', alpha=0.7, label='E-Divisive' if date==ediv_dates[0] else '')

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Which Method to Choose?

Based on the results:

- **PELT** provides exact segmentation with a clear cost function, making it ideal for offline analysis with known distribution assumptions.
- **BOCPD** gives probabilistic changepoint assessments and can process data sequentially, making it suitable for streaming applications and uncertainty quantification.
- **E-Divisive** makes no distribution assumptions, making it robust to different types of changes, including those beyond mean/variance shifts.

## Key Takeaways

1. **Different methods detect different types of changes**:
   - PELT focuses on changes in the specified cost function (mean/variance in our example)
   - BOCPD provides probabilistic assessments with uncertainty
   - E-Divisive captures distribution changes more broadly

2. **Parameter selection matters**:
   - PELT: Penalty controls sensitivity (BIC is more conservative than AIC)
   - BOCPD: Mean run length affects detection delay and false positive rate
   - E-Divisive: Minimum segment size and significance level affect detection

3. **Consider your application**:
   - For real-time monitoring: BOCPD
   - For precise offline segmentation: PELT
   - For distribution-free analysis: E-Divisive

## Next Steps

- Try these methods on your own time series data
- Explore other algorithms in ChangePointLab like kernel changepoint detection (KCP) or hidden semi-Markov models (HSMM)
- Check the documentation for advanced features like custom cost functions, adaptive hazard rates, and multivariate analysis

For more detailed examples, see the domain-specific tutorials on finance, healthcare, IoT, and environmental monitoring in the documentation.
