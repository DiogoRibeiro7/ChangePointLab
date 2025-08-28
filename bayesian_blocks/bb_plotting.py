# bb_plotting.py
# MIT License
# (c) 2025

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from bayesian_blocks import BBResult


def plot_blocks_time(
    *,
    t_min: float,
    t_max: float,
    result: BBResult,
    ax: Axes | None = None,
    title: str | None = "Bayesian Blocks (rate over time)",
) -> Axes:
    """
    Step plot of blockwise rate (Poisson) over time.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    edges = result.edges
    vals = result.block_value
    for i in range(len(vals)):
        ax.hlines(vals[i], edges[i], edges[i + 1])
        ax.vlines(edges[i], 0, vals[i], linestyles="dotted", linewidth=0.8)
    ax.vlines(edges[-1], 0, vals[-1], linestyles="dotted", linewidth=0.8)
    ax.set_xlim(t_min, t_max)
    ax.set_ylabel("rate")
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def plot_blocks_index(
    *,
    N: int,
    result: BBResult,
    ax: Axes | None = None,
    title: str | None = "Bayesian Blocks (per-index)",
    ylabel: str = "value",
) -> Axes:
    """
    Step plot of block values over integer index (counts/bernoulli settings).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    edges = result.edges.astype(int)
    vals = result.block_value
    for i in range(len(vals)):
        ax.hlines(vals[i], edges[i], edges[i + 1])
        ax.vlines(
            edges[i],
            min(vals.min(), 0),
            max(vals.max(), 0),
            linestyles="dotted",
            linewidth=0.8,
        )
    ax.vlines(
        edges[-1],
        min(vals.min(), 0),
        max(vals.max(), 0),
        linestyles="dotted",
        linewidth=0.8,
    )
    ax.set_xlim(0, N)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return ax
