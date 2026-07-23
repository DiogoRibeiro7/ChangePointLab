# edivisive_plotting.py
# MIT License
# (c) 2025

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ...algorithms.nonparametric.edivisive_core import EDivisiveResult


def plot_scree_edivisive(
    result: EDivisiveResult,
    *,
    ax: Axes | None = None,
    title: str | None = "E-Divisive: accepted split statistics",
) -> Axes:
    """
    Simple scree-style plot: bar chart of accepted test statistics (descending), annotated with p-values.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    if not result.splits:
        ax.text(
            0.5,
            0.5,
            "No changepoints accepted",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    stats = np.array([sp.statistic for sp in result.splits], dtype=float)
    pvals = [sp.pvalue for sp in result.splits]
    order = np.argsort(stats)[::-1]
    stats = stats[order]
    pvals = [pvals[i] for i in order]

    ax.bar(np.arange(stats.size), stats)
    for i, (s, p) in enumerate(zip(stats, pvals)):
        ax.text(i, s, f"p={p:.3f}", ha="center", va="bottom", fontsize=9, rotation=0)
    ax.set_xlabel("accepted splits (desc. by statistic)")
    ax.set_ylabel("test statistic")
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def plot_segments_1d(
    x: NDArray[np.floating],
    result: EDivisiveResult,
    *,
    ax: Axes | None = None,
    title: str | None = "E-Divisive segmentation (1D view)",
    lw: float = 1.25,
) -> Axes:
    """
    Quick 1D overlay: plot the signal x with vertical lines at accepted changepoints.
    For multivariate data, pass a 1-D projection (e.g., first PC) for visualization.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    n = x.size
    ax.plot(np.arange(n), x, linewidth=lw)
    for sp in result.splits:
        ax.axvline(sp.index, linestyle="--", linewidth=1.0)
    ax.set_xlim(0, n - 1)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return ax


__all__ = ["plot_scree_edivisive", "plot_segments_1d"]
