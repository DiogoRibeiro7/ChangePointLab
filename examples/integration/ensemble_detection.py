"""Ensemble changepoint detection with consensus clustering.

This example runs three algorithms (PELT, BOCPD and E-Divisive) on the same
signal and aggregates their changepoint proposals via simple clustering.  The
consensus locations can offer higher confidence than individual methods.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from changepoint_lab.algorithms.bayesian.bocpd import BOCPD, ConstantHazard
from pelt import pelt, NormalMeanVarUnknown
from changepoint_lab import edivisive
from examples.comparison_helpers import f1_score, plot_series


def generate_data(seed: int = 1):
    rng = np.random.default_rng(seed)
    segs = [rng.normal(0, 1, 80), rng.normal(0, 4, 60), rng.normal(2, 2, 100)]
    data = np.concatenate(segs)
    truth = np.cumsum([len(s) for s in segs])[:-1]
    return data, truth


def cluster_cps(cps_lists: list[list[int]], tol: int = 5):
    """Cluster changepoint proposals within ``tol`` samples."""
    all_cps = sorted(cp for cps in cps_lists for cp in cps)
    clusters: list[list[int]] = []
    for cp in all_cps:
        if not clusters or cp - clusters[-1][-1] > tol:
            clusters.append([cp])
        else:
            clusters[-1].append(cp)
    consensus = [int(np.mean(c)) for c in clusters]
    confidence = [len(c) / len(cps_lists) for c in clusters]
    return consensus, confidence


def main():
    data, truth = generate_data()

    hazard = ConstantHazard(mean_run_length=50)
    bocpd_model = BOCPD(hazard=hazard, alpha=1.0, beta=1.0)
    cps_bocpd = bocpd_model.fit_predict(data)
    cps_pelt = pelt(data, NormalMeanVarUnknown(), penalty=8.0)
    cps_ediv = edivisive(data, alpha=1.0)[0]

    consensus, conf = cluster_cps([cps_bocpd, cps_pelt, cps_ediv])
    metrics = {
        "BOCPD": f1_score(cps_bocpd, truth),
        "PELT": f1_score(cps_pelt, truth),
        "E-Divisive": f1_score(cps_ediv, truth),
        "Ensemble": f1_score(consensus, truth),
    }
    for name, m in metrics.items():
        print(f"{name:10s} F1={m['f1']:.2f} P={m['precision']:.2f} R={m['recall']:.2f}")

    fig, ax = plt.subplots(figsize=(8, 3))
    plot_series(ax, data, consensus, truth, label="Consensus")
    for cp, c in zip(consensus, conf):
        ax.text(cp, data.min(), f"{c:.1f}", color="red")
    ax.set_title("Consensus changepoints with confidence")
    plt.show()

    print("Consensus detection works well when multiple methods agree on a change;"
          " confidence scores highlight uncertain regions.")


if __name__ == "__main__":
    main()
