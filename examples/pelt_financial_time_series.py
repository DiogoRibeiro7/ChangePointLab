"""
PELT for Financial Time Series
==============================

Detect regime changes in a stock price series using PELT with a normal
cost function. Real prices are fetched with `yfinance` when available;
otherwise a synthetic geometric Brownian motion with volatility shifts is used.
Results are compared with BOCPD.
"""

import numpy as np
import matplotlib.pyplot as plt

from pelt import pelt, NormalMeanVarUnknown
from changepoint_lab.algorithms.bayesian.bocpd import BOCPD, ConstantHazard


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_prices():
    """Fetch SPY prices or generate synthetic data if download fails."""
    try:
        import yfinance as yf  # type: ignore
        df = yf.download("SPY", period="5y", progress=False)
        prices = df["Adj Close"].dropna().values
        print("Loaded real SPY prices")
    except Exception:
        rng = np.random.default_rng(0)
        n = 500
        mu = [0.0005, -0.0002, 0.0008]
        sigma = [0.01, 0.03, 0.015]
        lengths = [150, 200, 150]
        returns = np.concatenate([
            rng.normal(m, s, l) for m, s, l in zip(mu, sigma, lengths)
        ])
        prices = 100 * np.exp(np.cumsum(returns))
        print("Using synthetic price series")
    return prices


# ---------------------------------------------------------------------------
# PELT segmentation
# ---------------------------------------------------------------------------

def run_pelt(prices: np.ndarray):
    returns = np.diff(prices)
    penalty = 3 * np.log(len(returns))
    cps = pelt(returns, cost=NormalMeanVarUnknown(), penalty=penalty)
    return cps


# ---------------------------------------------------------------------------
# BOCPD comparison
# ---------------------------------------------------------------------------

def run_bocpd(prices: np.ndarray):
    returns = np.diff(prices)
    model = BOCPD(hazard=ConstantHazard(200), alpha=1.0, beta=1.0)
    cps = []
    for t, x in enumerate(returns, start=1):
        if model.update(x) > 0.4:
            cps.append(t)
    return cps


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def main():
    prices = load_prices()
    pelt_cps = run_pelt(prices)
    bocpd_cps = run_bocpd(prices)

    plt.figure(figsize=(10, 4))
    plt.plot(prices, label="Price")
    for cp in pelt_cps:
        plt.axvline(cp, color="r", alpha=0.6, label="PELT" if cp == pelt_cps[0] else "")
    for cp in bocpd_cps:
        plt.axvline(cp, color="g", linestyle="--", alpha=0.6, label="BOCPD" if cp == bocpd_cps[0] else "")
    plt.legend()
    plt.title("Financial time series segmentation")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()

    print("PELT changepoints:", pelt_cps)
    print("BOCPD changepoints:", bocpd_cps)


if __name__ == "__main__":
    main()
