"""
BOCPD for Activity Monitoring
=============================

This example demonstrates Bayesian Online Changepoint Detection (BOCPD)
applied to binary smart-home sensor data. A custom time-of-day hazard
boosts changepoint probability around daily boundaries. We compare
online BOCPD detection with an offline PELT segmentation.
"""

import numpy as np
import matplotlib.pyplot as plt

from changepoint_lab.algorithms.bayesian.bocpd import (
    BOCPD,
    BoostedBoundaryHazard,
)
from changepoint_lab.algorithms.optimization.pelt import BinomialCost, pelt


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_sensor_data(seed: int = 0):
    """Simulate daily activity with four regimes."""
    rng = np.random.default_rng(seed)
    probs = [0.05, 0.6, 0.2, 0.7]
    lengths = [50, 70, 40, 60]
    data = np.concatenate([rng.binomial(1, p, l) for p, l in zip(probs, lengths)])
    changepoints = np.cumsum(lengths)[:-1]
    return data, changepoints


# ---------------------------------------------------------------------------
# BOCPD with time-of-day hazard
# ---------------------------------------------------------------------------

def run_bocpd(data: np.ndarray):
    """Run BOCPD online and return detected changepoints."""
    # mean_run_length reflects expected regime duration (~50 samples)
    hazard = BoostedBoundaryHazard(mean_run_length=50, boundaries=np.arange(0, len(data), 50))
    model = BOCPD(hazard=hazard, alpha=1.0, beta=1.0)
    cps = []
    for t, x in enumerate(data):
        if model.update(x) > 0.5:  # high posterior changepoint probability
            cps.append(t)
    return cps


# ---------------------------------------------------------------------------
# PELT comparison
# ---------------------------------------------------------------------------

def run_pelt(data: np.ndarray):
    """Offline segmentation using PELT."""
    penalty = 2 * np.log(len(data))
    cps = pelt(data, cost=BinomialCost(), penalty=penalty)
    return cps


# ---------------------------------------------------------------------------
# Visualization and comparison
# ---------------------------------------------------------------------------

def main():
    data, true_cps = generate_sensor_data()
    bocpd_cps = run_bocpd(data)
    pelt_cps = run_pelt(data)

    plt.figure(figsize=(10, 4))
    plt.step(range(len(data)), data, where="post", label="sensor")
    for cp in true_cps:
        plt.axvline(cp, color="k", linestyle="--", alpha=0.3)
    for cp in bocpd_cps:
        plt.axvline(cp, color="r", linestyle="-", alpha=0.7, label="BOCPD" if cp == bocpd_cps[0] else "")
    for cp in pelt_cps:
        plt.axvline(cp, color="b", linestyle="-.", alpha=0.7, label="PELT" if cp == pelt_cps[0] else "")
    plt.legend()
    plt.title("Activity monitoring with BOCPD vs PELT")
    plt.xlabel("Time")
    plt.ylabel("Motion")
    plt.tight_layout()
    plt.show()

    print("True changepoints:", true_cps)
    print("BOCPD detected:", bocpd_cps)
    print("PELT detected:", pelt_cps)


if __name__ == "__main__":
    main()
