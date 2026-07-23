# bocpd_plotting.py
# MIT License
# (c) 2025

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Any

from changepoint_lab._optional import require_matplotlib_pyplot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any


def plot_run_length_heatmap(
    R: NDArray[np.floating],
    *,
    vmax: float | None = None,
    ax: Axes | None = None,
    title: str | None = "BOCPD run-length posterior",
) -> Axes:
    """
    Heatmap for run-length posterior over time.

    Parameters
    ----------
    R : array (T, R+1)
        Stored run-length posterior (row t holds P(r_t = r | x_{1:t})).
    vmax : Optional[float]
        Upper bound for color scaling; by default uses max in R.
    ax : Optional[Axes]
        Destination axes.
    title : Optional[str]
        Title to set (None to disable).
    """
    if R.ndim != 2:
        raise ValueError("R must be 2-D (T x (R+1)).")
    plt = require_matplotlib_pyplot("BOCPD plotting")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(R.T, aspect="auto", origin="lower", interpolation="nearest", vmax=vmax)
    ax.set_xlabel("time t")
    ax.set_ylabel("run length r")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return ax


def plot_cp_probability(
    cp_prob: NDArray[np.floating],
    *,
    ax: Axes | None = None,
    title: str | None = "Changepoint probability (P[r_t=0])",
) -> Axes:
    """
    Line plot for P(r_t=0 | x_{1:t}).

    Parameters
    ----------
    cp_prob : array (T,)
        CP probability per time step.
    ax : Optional[Axes]
        Destination axes.
    title : Optional[str]
        Title to set (None to disable).
    """
    if cp_prob.ndim != 1:
        raise ValueError("cp_prob must be 1-D.")
    plt = require_matplotlib_pyplot("BOCPD plotting")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(np.arange(cp_prob.size), cp_prob, linewidth=1.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("time t")
    ax.set_ylabel("P(CP)")
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return ax
