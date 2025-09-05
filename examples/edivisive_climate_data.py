"""
E-Divisive for Climate Data
===========================

Analyze multivariate climate indicators (temperature and precipitation)
using the nonparametric E-Divisive method with permutation testing.
Results are compared with a Gaussian PELT segmentation.
"""

import numpy as np
import matplotlib.pyplot as plt

from changepoint_lab import edivisive
from pelt import pelt, NormalMeanVarUnknown


# ---------------------------------------------------------------------------
# Synthetic climate generation
# ---------------------------------------------------------------------------

def generate_climate(seed: int = 0):
    rng = np.random.default_rng(seed)
    lengths = [300, 300, 300]
    temps = [15.0, 16.5, 14.0]
    precs = [5.0, 7.0, 4.0]
    data = np.vstack([
        np.column_stack([
            rng.normal(t, 0.5, l),
            rng.normal(p, 0.3, l)
        ])
        for t, p, l in zip(temps, precs, lengths)
    ])
    cps = np.cumsum(lengths)[:-1]
    return data, cps


# ---------------------------------------------------------------------------
# Detection methods
# ---------------------------------------------------------------------------

def run_edivisive(data: np.ndarray):
    res = edivisive(data, alpha=1.0, R=99, min_size=20)
    return res.cps


def run_pelt(data: np.ndarray):
    penalty = 4 * np.log(len(data))
    cps = pelt(data[:, 0], cost=NormalMeanVarUnknown(), penalty=penalty)
    return cps


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def main():
    data, true_cps = generate_climate()
    cps_ediv = run_edivisive(data)
    cps_pelt = run_pelt(data)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(data[:, 0], label="Temperature")
    ax[1].plot(data[:, 1], label="Precipitation", color="tab:orange")
    for cp in true_cps:
        for a in ax:
            a.axvline(cp, color="k", linestyle="--", alpha=0.3)
    for cp in cps_ediv:
        for a in ax:
            a.axvline(cp, color="r", alpha=0.7, label="E-Divisive" if a is ax[0] and cp == cps_ediv[0] else "")
    for cp in cps_pelt:
        for a in ax:
            a.axvline(cp, color="b", linestyle="-.", alpha=0.7, label="PELT" if a is ax[0] and cp == cps_pelt[0] else "")
    for a in ax:
        a.legend()
    ax[1].set_xlabel("Time")
    ax[0].set_ylabel("Temp")
    ax[1].set_ylabel("Precip")
    plt.tight_layout()
    plt.show()

    print("True changepoints:", true_cps)
    print("E-Divisive detected:", cps_ediv)
    print("PELT detected:", cps_pelt)


if __name__ == "__main__":
    main()
