"""Hierarchical changepoint analysis using PELT and E-Divisive.

Major regime changes are detected first with PELT.  Each resulting segment is
then analysed with E-Divisive to uncover sub-changepoints, yielding a hierarchy
of structural breaks.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from changepoint_lab.algorithms.optimization.pelt import (
    NormalMeanVarUnknown,
    pelt,
)
from changepoint_lab import edivisive
from examples.comparison_helpers import plot_series


def generate_data(seed: int = 2):
    rng = np.random.default_rng(seed)
    major1 = rng.normal(0, 1, 150)
    major2 = rng.normal(5, 1, 150)
    # each major block has an internal shift
    major1[75:] += 2
    major2[80:] -= 3
    data = np.concatenate([major1, major2])
    major_cps = [150]
    sub_cps = [75, 230]
    return data, major_cps, sub_cps


def main():
    data, major_truth, sub_truth = generate_data()

    # Detect major changepoints
    major_cps = pelt(data, NormalMeanVarUnknown(), penalty=20.0)

    hierarchy = {"major": major_cps, "minor": []}
    last = 0
    for cp in major_cps + [len(data)]:
        seg = data[last:cp]
        if len(seg) > 5:
            minor, _ = edivisive(seg, alpha=1.0)
            hierarchy["minor"].extend(last + np.array(minor, dtype=int))
        last = cp

    # Plot hierarchy
    fig, ax = plt.subplots(figsize=(8, 3))
    plot_series(ax, data, hierarchy["minor"], major_truth + sub_truth, label="Minor within major")
    for cp in hierarchy["major"]:
        ax.axvline(cp, color="blue", linewidth=2, label="Major" if cp == hierarchy["major"][0] else None)
    ax.set_title("Hierarchical changepoints")
    plt.show()

    print("Major changepoints (PELT):", hierarchy["major"])
    print("Minor changepoints (E-Divisive):", hierarchy["minor"])
    print("Hierarchical analysis is useful when changes occur at multiple scales.")


if __name__ == "__main__":
    main()
