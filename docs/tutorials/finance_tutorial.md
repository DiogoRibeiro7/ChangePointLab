# Financial Analysis Tutorial

This tutorial shows how changepoint detection assists in financial time‑series analysis for regime shifts and volatility changes.

## Market Regime Detection
- Obtain adjusted close prices for indices or equities (e.g., via `yfinance`)
- Compute log returns and realized volatility

```python
import yfinance as yf
prices = yf.download("SPY", start="2015-01-01")['Adj Close']
returns = prices.pct_change().dropna()
```

## Volatility Changepoint Analysis
- **PELT** with a Gaussian cost on log returns to find volatility regimes
- **BOCPD** only after transforming the problem to a supported binary or count
  stream; Student-t BOCPD is not currently implemented
- **E‑Divisive** to detect nonparametric shifts across multiple assets

## Portfolio Rebalancing Based on Detected Regimes
1. Segment returns using PELT
2. Estimate mean/variance per segment
3. Rebalance portfolio weights when a new regime is confirmed

## High‑Frequency Data Considerations
- Downsample to manageable frequency to reduce microstructure noise
- Use rolling window BOCPD for latency‑sensitive trading

## Case Study: Trading Strategy Adaptation
- Dataset: 5‑minute SPY returns for 3 months
- Strategy: Switch between momentum and mean‑reversion after changepoint detection
- Metrics: Sharpe ratio and turnover before/after adaptation

## Parameter Tuning Considerations
- Penalty term for PELT based on BIC with `log(n)` scaling
- Hazard function for BOCPD set to mean run length of typical regime duration (e.g., 20 days)
- Minimum segment length to avoid reacting to noise (e.g., 10 bars)

## Evaluation Metrics
- Return/risk trade‑off improvements
- Regime classification accuracy against known market events
- Latency from actual market shift to detection

## Interpretation Guidelines
- Align detected regimes with macro events (e.g., policy changes)
- Evaluate transaction costs when rebalancing on changepoints

## Complete Workflow Example
See `examples/pelt_financial_time_series.py` for a full example.

## References
- L. Cao et al., "Financial Regime Detection using Changepoint Analysis," 2019.
- P. Fearnhead, "Exact Bayesian Curve Fitting and Signal Segmentation," 2006.
