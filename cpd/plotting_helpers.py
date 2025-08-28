# plotting_helpers.py
# MIT License
# (c) 2025

from __future__ import annotations

from typing import Optional, Sequence, Tuple, TypedDict

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Reuse the Tau type alias from your model file if you prefer:
Tau = Tuple[int, ...]


class PWSummary(TypedDict):
    """Pointwise posterior summary container."""
    median: NDArray[np.floating]
    lower: NDArray[np.floating]
    upper: NDArray[np.floating]


# ------------------------- Tick helpers (optional) -------------------------

def _apply_time_of_day_ticks(
    ax: Axes,
    N: int,
    *,
    start_hour: int = 0,
    hours_step: int = 6,
) -> None:
    """
    Put "HH:00" ticks every `hours_step` hours if N is divisible by 24.
    Falls back to index ticks if not cleanly divisible.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to modify.
    N : int
        Number of bins in the daily period.
    start_hour : int, default=0
        Time-of-day (0-23) corresponding to index 0 on the x-axis.
    hours_step : int, default=6
        Tick every `hours_step` hours.
    """
    if N % 24 != 0:
        # Not evenly spaced by hour; skip fancy ticks
        ax.set_xlabel("Time-of-day index (0..N-1)")
        return

    bins_per_hour = N // 24
    tick_hours = list(range(0, 25, hours_step))  # include 24
    tick_pos = [h * bins_per_hour for h in tick_hours]
    tick_labels = [f"{(start_hour + h) % 24:02d}:00" for h in tick_hours]
    ax.set_xticks(tick_pos, tick_labels, rotation=0)
    ax.set_xlabel("Time of day")


# ------------------------- Plot 1: CP posterior mass -------------------------

def plot_changepoint_posterior_mass(
    *,
    cp_hist: NDArray[np.integer],
    num_samples: int,
    N: int,
    tau_map: Optional[Tau] = None,
    normalize: str = "per-sample",
    start_hour: int = 0,
    hours_step: int = 6,
    ax: Optional[Axes] = None,
    title: Optional[str] = "Changepoint posterior mass",
) -> Axes:
    """
    Plot the posterior probability of a changepoint at each time-of-day index.

    Parameters
    ----------
    cp_hist : array-like of shape (N,)
        Histogram of changepoint occurrences across kept MCMC samples.
        Each position counts how many samples contained a CP at that index.
    num_samples : int
        Number of kept samples used to build `cp_hist`.
    N : int
        Period length.
    tau_map : Optional[Tau], default=None
        Optional MAP changepoint set to overlay as vertical lines.
    normalize : {"per-sample", "sum-1"}, default="per-sample"
        If "per-sample": plot cp_hist / num_samples (posterior P[CP at r]).
        If "sum-1": normalize so the bars sum to 1.
    start_hour : int, default=0
        Time-of-day at index 0. Used only for ticks when N divisible by 24.
    hours_step : int, default=6
        Tick every `hours_step` hours when applicable.
    ax : Optional[Axes], default=None
        If provided, draw on this axes; otherwise create a new figure.
    title : Optional[str], default="Changepoint posterior mass"
        Title to use; pass None to disable.

    Returns
    -------
    Axes
        The matplotlib Axes with the plot.
    """
    cp_hist = np.asarray(cp_hist, dtype=float)
    if cp_hist.ndim != 1 or cp_hist.size != N:
        raise ValueError("cp_hist must be 1-D with length N.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    if normalize == "per-sample":
        y = cp_hist / float(num_samples)
        ylabel = "Posterior P(change at r)"
    elif normalize == "sum-1":
        s = float(cp_hist.sum())
        y = cp_hist / s if s > 0 else cp_hist
        ylabel = "Normalized mass (sums to 1)"
    else:
        raise ValueError("normalize must be 'per-sample' or 'sum-1'.")

    x = np.arange(N)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.2))

    # Use a thin bar plot for robustness (avoids stem() compatibility quirks).
    ax.bar(x, y, width=1.0, align="center")
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Optional overlay of MAP changepoints
    if tau_map:
        for cp in tau_map:
            ax.axvline(cp, linestyle="--")  # default color/style only

    _apply_time_of_day_ticks(ax, N, start_hour=start_hour, hours_step=hours_step)
    ax.grid(True, axis="y", alpha=0.3)  # light grid for readability
    return ax


# ------------------------- Plot 2: Pointwise bands -------------------------

def plot_pointwise_bands(
    *,
    pw: PWSummary,
    tau: Optional[Tau] = None,
    start_hour: int = 0,
    hours_step: int = 6,
    ax: Optional[Axes] = None,
    title: Optional[str] = "Pointwise posterior bands",
) -> Axes:
    """
    Plot median and credible band for p(t) on the N-lattice.

    Parameters
    ----------
    pw : dict with keys {"median","lower","upper"}
        Output of WithinPeriodCPD.pointwise_posterior_summary_from_samples(...).
        All arrays must be shape (N,).
    tau : Optional[Tau], default=None
        Optional changepoints to overlay as vertical lines.
    start_hour : int, default=0
        Time-of-day at index 0. Used only for ticks when N divisible by 24.
    hours_step : int, default=6
        Tick every `hours_step` hours when applicable.
    ax : Optional[Axes], default=None
        If provided, draw on this axes; otherwise create a new figure.
    title : Optional[str], default="Pointwise posterior bands"
        Title to use; pass None to disable.

    Returns
    -------
    Axes
        The matplotlib Axes with the plot.
    """
    median = np.asarray(pw["median"], dtype=float)
    lower = np.asarray(pw["lower"], dtype=float)
    upper = np.asarray(pw["upper"], dtype=float)

    if median.shape != lower.shape or median.shape != upper.shape or median.ndim != 1:
        raise ValueError("pw['median'], ['lower'], ['upper'] must be 1-D arrays of equal shape.")
    N = median.size
    x = np.arange(N)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.2))

    # Band first so line sits on top
    ax.fill_between(x, lower, upper, alpha=0.3, linewidth=0)
    ax.plot(x, median, linewidth=1.5)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylabel("p(t)")
    if title:
        ax.set_title(title)

    # Optional overlay of changepoints (e.g., MAP)
    if tau:
        for cp in tau:
            ax.axvline(cp, linestyle="--")

    _apply_time_of_day_ticks(ax, N, start_hour=start_hour, hours_step=hours_step)
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def plot_posterior_num_segments(
    m_values: Sequence[int],
    probs: Sequence[float],
    ax: Axes | None = None,
    title: str | None = "Posterior over number of segments (m)",
) -> Axes:
    """
    Bar plot of posterior mass over m.

    Parameters
    ----------
    m_values : sequence of ints
        Unique m values (sorted).
    probs : sequence of floats
        Normalized probabilities for each m.
    ax : Optional[Axes], default=None
        If provided, draw on this axes; otherwise create a new figure.
    title : Optional[str], default=...
        Title to use; pass None to disable.
    """
    m_values = np.asarray(m_values, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if m_values.size != probs.size:
        raise ValueError("m_values and probs must have the same length.")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    ax.bar(m_values, probs, width=0.8)
    ax.set_xlabel("m")
    ax.set_ylabel("Posterior mass")
    if title:
        ax.set_title(title)
    ax.set_xlim(m_values.min() - 0.9, m_values.max() + 0.9)
    ax.grid(True, axis="y", alpha=0.3)
    return ax

# # after running the MCMC:
# from plotting_helpers import plot_changepoint_posterior_mass, plot_pointwise_bands

# # 1) Changepoint posterior mass
# ax1 = plot_changepoint_posterior_mass(
#     cp_hist=result.changepoint_hist,
#     num_samples=len(result.samples_tau),
#     N=prior.N,
#     tau_map=result.mode_tau,        # optional overlay
#     start_hour=0,                   # set if index 0 == 00:00
#     hours_step=6,                   # ticks every 6 hours (requires N % 24 == 0)
# )

# # 2) Pointwise posterior bands (median + CI)
# pw = model.pointwise_posterior_summary_from_samples(
#     result.samples_tau,
#     draws_per_sample=2,
#     credible=0.95,
# )
# ax2 = plot_pointwise_bands(
#     pw=pw,
#     tau=result.mode_tau,            # optional overlay
#     start_hour=0,
#     hours_step=6,
# )

# plt.show()


# from diagnostics import posterior_num_segments
# from plotting_helpers import plot_posterior_num_segments

# pm = posterior_num_segments(result.samples_tau)
# plot_posterior_num_segments(pm.m_values, pm.probs)
