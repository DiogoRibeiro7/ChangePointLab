# kcp_plotting.py
# MIT License
# (c) 2025

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Any

from changepoint_lab._optional import require_matplotlib_pyplot

from ...algorithms.kernel.kcp_core import KCPResult, KCPModelSel

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any


def plot_segments_1d(
    x: NDArray[np.floating],
    edges: NDArray[np.integer],
    *,
    ax: Axes | None = None,
    title: str | None = "Kernel CPD segmentation (1D view)",
) -> Axes:
    """
    Overlay a 1D series with vertical lines at fitted edges.
    For multivariate X, pass a 1D projection (e.g., first PC) here.
    """
    plt = require_matplotlib_pyplot("KCP plotting")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    n = x.size
    ax.plot(np.arange(n), x, linewidth=1.25)
    for e in edges[1:-1]:
        ax.axvline(int(e), linestyle="--", linewidth=1.0)
    ax.set_xlim(0, n - 1)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def plot_model_scree(
    sel: KCPModelSel,
    *,
    ax: Axes | None = None,
    title: str | None = "KCP: cost vs. #segments (BIC-style)",
) -> Axes:
    """
    Scree plot of unpenalized costs and penalized criterion across m.
    """
    plt = require_matplotlib_pyplot("KCP plotting")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    m = np.arange(1, sel.costs_m.size + 1)
    ax.plot(m, sel.costs_m, marker="o", label="unpenalized cost")
    ax.plot(m, sel.penalized_m, marker="o", linestyle="--", label=f"penalized (beta={sel.beta})")
    ax.axvline(sel.m_star, linestyle=":", linewidth=1.0)
    ax.set_xlabel("#segments (m)")
    ax.set_ylabel("objective")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return ax
