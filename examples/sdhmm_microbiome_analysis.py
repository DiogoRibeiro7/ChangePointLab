"""
SD-HMM for Microbiome Analysis
==============================

Demonstrate the Scaled-Dirichlet HMM on compositional microbiome
abundance data. We compare SD-HMM segmentation with an E-Divisive
analysis on centered log-ratio (clr) transformed data.
"""

import numpy as np
import matplotlib.pyplot as plt

from changepoint_lab import SDHMM, SDHMMConfig
from changepoint_lab import edivisive


# ---------------------------------------------------------------------------
# Synthetic compositional data
# ---------------------------------------------------------------------------

def generate_microbiome(seed: int = 0):
    rng = np.random.default_rng(seed)
    lengths = [200, 300, 250]
    alphas = [
        np.array([5, 1, 1, 1]),
        np.array([1, 5, 1, 1]),
        np.array([1, 1, 5, 1]),
    ]
    data = np.vstack([rng.dirichlet(a, l) for a, l in zip(alphas, lengths)])
    cps = np.cumsum(lengths)[:-1]
    return data, cps


# ---------------------------------------------------------------------------
# SD-HMM fit
# ---------------------------------------------------------------------------

def run_sdhmm(data: np.ndarray):
    model = SDHMM(SDHMMConfig(K=3, max_iter=100, min_iter=5, tol=1e-5))
    model.fit(data)
    z = model.viterbi(data)
    cps = np.where(np.diff(z) != 0)[0] + 1
    return cps


# ---------------------------------------------------------------------------
# E-Divisive on clr-transformed data
# ---------------------------------------------------------------------------

def run_edivisive(data: np.ndarray):
    clr = np.log(data) - np.mean(np.log(data), axis=1, keepdims=True)
    res = edivisive(clr, alpha=1.0, R=49)
    return res.cps


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def main():
    data, true_cps = generate_microbiome()
    cps_sdhmm = run_sdhmm(data)
    cps_ediv = run_edivisive(data)

    plt.figure(figsize=(10, 4))
    plt.stackplot(range(len(data)), data.T, labels=[f"sp{i}" for i in range(data.shape[1])])
    for cp in true_cps:
        plt.axvline(cp, color="k", linestyle="--", alpha=0.3)
    for cp in cps_sdhmm:
        plt.axvline(cp, color="r", alpha=0.7, label="SD-HMM" if cp == cps_sdhmm[0] else "")
    for cp in cps_ediv:
        plt.axvline(cp, color="b", linestyle="-.", alpha=0.7, label="E-Divisive" if cp == cps_ediv[0] else "")
    plt.legend(loc="upper right")
    plt.title("Microbiome compositional changepoints")
    plt.xlabel("Time")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.show()

    print("True changepoints:", true_cps)
    print("SD-HMM detected:", cps_sdhmm)
    print("E-Divisive detected:", cps_ediv)


if __name__ == "__main__":
    main()
