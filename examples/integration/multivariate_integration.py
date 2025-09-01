"""Combining univariate and multivariate changepoint detection.

A multivariate method (E-Divisive) captures joint shifts across channels while
univariate PELT runs on each channel to find local deviations.  Comparing the two
helps distinguish global from channel-specific changes.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from edivisive import edivisive
from pelt import pelt, NormalMeanVarUnknown
from examples.comparison_helpers import plot_series


def generate_data(seed: int = 3):
    rng = np.random.default_rng(seed)
    n = 200
    x = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(3, 1, n // 2)])
    y = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(0, 2, n // 2)])
    # channel-specific change in y near the end
    y[150:] += 4
    data = np.vstack([x, y]).T
    global_cp = [n // 2]
    local_cp = [150]
    return data, global_cp, local_cp


def main():
    data, global_truth, local_truth = generate_data()

    # Multivariate detection
    mv_cps, _ = edivisive(data, alpha=1.0)

    # Univariate detection per channel
    uni_cps = []
    for channel in data.T:
        cps = pelt(channel, NormalMeanVarUnknown(), penalty=10.0)
        uni_cps.append(cps)
    union_cps = sorted(set(c for cps in uni_cps for c in cps))

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    plot_series(axes[0], data[:, 0], mv_cps, global_truth, label="E-Divisive on X")
    plot_series(axes[1], data[:, 1], union_cps, global_truth + local_truth, label="PELT on Y")
    axes[0].set_title("Global vs local changepoints")
    plt.show()

    print("Global changepoints (multivariate):", mv_cps)
    print("Local changepoints (per-channel):", uni_cps)
    print("Multivariate detection captures joint shifts, while per-channel PELT"
          " reveals channel-specific anomalies.")


if __name__ == "__main__":
    main()
