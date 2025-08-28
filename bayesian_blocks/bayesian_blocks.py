# bayesian_blocks.py
# MIT License
# (c) 2025

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import math
import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.floating]
ArrayI = NDArray[np.integer]


# ---------------------------------------------------------------------
# Priors / penalties
# ---------------------------------------------------------------------

def ncp_prior_from_p0(n: int, p0: float = 0.05) -> float:
    """
    Nonconformity (changepoint) prior for Bayesian Blocks from Scargle (2013),
    calibrated by the desired overall false positive rate p0 for a signal with n cells.

    Returns
    -------
    gamma : float
        Additive penalty per block (i.e., subtracted once per block in the fitness DP).

    Notes
    -----
    This is the common analytic approximation used in practice (e.g., astropy):
        gamma = 4 - log(73.53 * p0 * n**(-0.478))
    where log is natural log.
    """
    if not (0 < p0 < 1):
        raise ValueError("p0 must be in (0,1).")
    if n < 1:
        raise ValueError("n must be >= 1.")
    return 4.0 - math.log(73.53 * p0 * (n ** -0.478))


# ---------------------------------------------------------------------
# Core DP solver
# ---------------------------------------------------------------------

@dataclass
class BBResult:
    """Result of a Bayesian Blocks run."""
    # For time-based problems (events or binned counts), edges are increasing breakpoints (including start & end).
    # For index-based (Bernoulli over samples), edges are integer indices [0, ..., N].
    edges: ArrayF
    # Per-block MLE parameter (rate for Poisson; probability for Bernoulli) in each block.
    block_value: ArrayF
    # Indices of change points in the input 'cells' space (exclusive right edge).
    change_points: ArrayI


def _dp_solve(
    *,
    # cumulative sufficient statistics over "cells" 0..N-1
    # These must be prefix sums so a block [i, j) uses sums[j] - sums[i].
    stat_num: ArrayF,   # numerator-like (e.g., counts, successes)
    stat_den: ArrayF,   # denominator-like (e.g., exposure, trials); for Bernoulli this is "trials"
    fitness_per_block: Callable[[float, float], float],  # f(num, den) -> scalar
    gamma: float,
) -> Tuple[ArrayI, ArrayI, ArrayF]:
    """
    Generic O(N^2) dynamic program for Bayesian Blocks.

    Parameters
    ----------
    stat_num : prefix sum of numerators, shape (N+1,)
    stat_den : prefix sum of denominators, shape (N+1,)
    fitness_per_block : function on block totals (num, den)
    gamma : additive penalty per block

    Returns
    -------
    last : ArrayI
        Backpointers: last[j] = argmax i
    cp : ArrayI
        List of changepoints in cell index (right-exclusive), reconstructed.
    opt : ArrayF
        DP objective F[j] (max fitness up to cell j)
    """
    Np = int(stat_num.size)
    if stat_den.size != Np:
        raise ValueError("stat_num and stat_den must have the same length (prefix arrays).")
    if Np < 2:
        # nothing to segment
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.zeros(1, dtype=float)

    N = Np - 1  # number of cells

    # DP arrays: opt[j] = best fitness up to j; last[j] = argmax i for j
    opt = np.empty(N + 1, dtype=float)
    last = np.empty(N + 1, dtype=np.int64)
    # base case: no cells -> 0 (we apply -gamma inside transitions so m*gamma overall)
    opt[0] = 0.0
    last[0] = -1

    # Pre-allocate for speed
    idx = np.arange(N, dtype=np.int64)

    for j in range(1, N + 1):
        # Try all possible starts i in [0, j)
        # Block totals:
        num = stat_num[j] - stat_num[:j]  # shape (j,)
        den = stat_den[j] - stat_den[:j]  # shape (j,)
        # Fitness for each candidate block [i, j)
        fit = np.fromiter((fitness_per_block(nu, de) for nu, de in zip(num, den)), count=j, dtype=float)
        # Total objective if last change at i: opt[i] + fit(i->j) - gamma
        total = opt[:j] + fit - gamma
        i_star = int(np.argmax(total))
        opt[j] = float(total[i_star])
        last[j] = i_star

    # Reconstruct change points by backtracking
    cps: List[int] = []
    j = N
    while j > 0:
        i = int(last[j])
        if i == 0:
            cps.append(j)
            break
        cps.append(j)
        j = i
    cps = list(reversed(cps))

    return last, np.asarray(cps, dtype=np.int64), opt


# ---------------------------------------------------------------------
# Fitness functions
# ---------------------------------------------------------------------

def _fit_poisson(num: float, den: float) -> float:
    """
    Poisson process / counts:
      num = total counts in block (k),
      den = total exposure/width in block (T),
      fitness = k * (log k - log T), with convention 0*log(0/T) := 0.
    """
    if den <= 0:
        return -np.inf
    if num <= 0:
        return 0.0
    return num * (math.log(num) - math.log(den))


def _fit_bernoulli(success: float, trials: float) -> float:
    """
    Bernoulli/Binomial:
      success = # successes in block (s),
      trials  = # trials in block (n >= s),
      fitness = s*log(s/n) + (n-s)*log(1 - s/n), with 0*log 0 := 0.

    Note: we omit the binomial coefficient log C(n, s) since it’s constant w.r.t the
    parameter and standard in Bayesian Blocks to drop parameter-independent terms.
    """
    if trials <= 0:
        return -np.inf
    s = success
    n = trials
    if s <= 0 or s >= n:
        # handle edge cases; use limits s->0 or s->n
        if s <= 0:
            return n * math.log(max(1.0 - 1e-16, 1.0))  # -> 0
        else:
            return n * math.log(1e-16)  # ~ -inf, but this path is uncommon
    p = s / n
    return s * math.log(p) + (n - s) * math.log(1.0 - p)


# ---------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------

def bayesian_blocks_events(
    t: Sequence[float],
    *,
    t_start: Optional[float] = None,
    t_stop: Optional[float] = None,
    p0: Optional[float] = 0.05,
    gamma: Optional[float] = None,
) -> BBResult:
    """
    Bayesian Blocks for **event times** (unbinned Poisson process).

    Parameters
    ----------
    t : sequence of floats
        Sorted or unsorted event timestamps.
    t_start, t_stop : optional floats
        Start and stop of observation window. If None, inferred from data using
        half-interval edges around the min/max event times.
    p0 : Optional[float], default=0.05
        Target false positive rate. If provided, overrides `gamma` via the Scargle prior.
    gamma : Optional[float]
        Direct penalty per block. Use either p0 or gamma (p0 takes precedence if both set).

    Returns
    -------
    BBResult with:
        edges: breakpoints in time (length = #blocks+1)
        block_value: MLE rate per block (events per unit time)
        change_points: indices in the *event-cell* space
    """
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        return BBResult(edges=np.array([], dtype=float),
                        block_value=np.array([], dtype=float),
                        change_points=np.array([], dtype=np.int64))
    t = np.sort(t)
    # Build event-cell edges (Voronoi): midpoints between events; clip/extend at t_start/t_stop.
    dt_prev = np.diff(t, prepend=(2 * t[0] - t[1]) if t.size > 1 else t[0] - 1.0)
    dt_next = np.diff(t, append=(2 * t[-1] - t[-2]) if t.size > 1 else t[-1] + 1.0)
    edges = np.empty(t.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (t[:-1] + t[1:])
    edges[0] = t[0] - 0.5 * (t[1] - t[0]) if t.size > 1 else t[0] - 0.5
    edges[-1] = t[-1] + 0.5 * (t[-1] - t[-2]) if t.size > 1 else t[-1] + 0.5
    if t_start is not None:
        edges[0] = float(t_start)
    if t_stop is not None:
        edges[-1] = float(t_stop)
    # Each event defines one "cell": count=1, exposure = cell width
    widths = np.diff(edges)                      # (N,)
    counts = np.ones(t.size, dtype=float)        # (N,)
    # Prefix sums
    K = np.concatenate([[0.0], np.cumsum(counts)])  # counts
    T = np.concatenate([[0.0], np.cumsum(widths)])  # exposures
    # Penalty
    g = ncp_prior_from_p0(len(counts), p0) if (p0 is not None) else float(gamma if gamma is not None else 0.0)
    # Solve
    last, cps, opt = _dp_solve(stat_num=K, stat_den=T, fitness_per_block=_fit_poisson, gamma=g)

    # Build final block edges and rates
    # convert cps (cell indices) to time edges
    cp_edges = np.concatenate([[edges[0]], edges[cps], [edges[-1]]])
    # Per-block MLE rates:
    block_counts = np.diff(np.concatenate([[0.0], K[cps], [K[-1]]]))
    block_exposure = np.diff(np.concatenate([[0.0], T[cps], [T[-1]]]))
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.where(block_exposure > 0, block_counts / block_exposure, 0.0)

    return BBResult(edges=cp_edges, block_value=rates, change_points=cps)


def bayesian_blocks_counts(
    counts: Sequence[float],
    widths: Optional[Sequence[float]] = None,
    *,
    p0: Optional[float] = 0.05,
    gamma: Optional[float] = None,
) -> BBResult:
    """
    Bayesian Blocks for **binned Poisson counts** with per-bin exposure/width.

    Parameters
    ----------
    counts : sequence (N,)
        Count in each bin (non-negative).
    widths : optional sequence (N,)
        Exposure/width for each bin (positive). If None, all ones.
    p0, gamma : as in bayesian_blocks_events (p0 takes precedence if set).

    Returns
    -------
    BBResult with:
        edges: integer bin edges [0..N]
        block_value: rate per unit exposure within each block
        change_points: bin indices (right-exclusive)
    """
    c = np.asarray(counts, dtype=float)
    if c.ndim != 1:
        raise ValueError("counts must be 1-D.")
    if np.any(c < 0):
        raise ValueError("counts must be non-negative.")
    N = c.size
    w = np.ones(N, dtype=float) if widths is None else np.asarray(widths, dtype=float)
    if w.shape != c.shape or np.any(w <= 0):
        raise ValueError("widths must match counts shape and be > 0.")
    K = np.concatenate([[0.0], np.cumsum(c)])
    T = np.concatenate([[0.0], np.cumsum(w)])
    g = ncp_prior_from_p0(N, p0) if (p0 is not None) else float(gamma if gamma is not None else 0.0)

    last, cps, opt = _dp_solve(stat_num=K, stat_den=T, fitness_per_block=_fit_poisson, gamma=g)

    # Build edges (bin indices) and block rates
    edges = np.array([0, *cps.tolist(), N], dtype=float)
    block_counts = np.diff(np.concatenate([[0.0], K[cps], [K[-1]]]))
    block_exposure = np.diff(np.concatenate([[0.0], T[cps], [T[-1]]]))
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.where(block_exposure > 0, block_counts / block_exposure, 0.0)

    return BBResult(edges=edges, block_value=rates, change_points=cps)


def bayesian_blocks_bernoulli(
    successes: Sequence[int] | Sequence[float],
    trials: Optional[Sequence[int] | Sequence[float]] = None,
    *,
    p0: Optional[float] = 0.05,
    gamma: Optional[float] = None,
) -> BBResult:
    """
    Bayesian Blocks for **Bernoulli/Binomial** data: successes out of trials per cell.

    Parameters
    ----------
    successes : sequence (N,)
        Number of successes in each cell (0..trials).
    trials : sequence (N,), optional
        Number of trials per cell (>0). If None, all ones (i.e., raw binary stream).
    p0, gamma : as before (p0 overrides gamma if set).

    Returns
    -------
    BBResult with:
        edges: integer cell edges [0..N]
        block_value: MLE success probability p̂ per block
        change_points: cell indices (right-exclusive)
    """
    s = np.asarray(successes, dtype=float)
    if s.ndim != 1:
        raise ValueError("successes must be 1-D.")
    if np.any(s < 0):
        raise ValueError("successes must be >= 0.")
    N = s.size
    n = np.ones(N, dtype=float) if trials is None else np.asarray(trials, dtype=float)
    if n.shape != s.shape or np.any(n <= 0) or np.any(s > n):
        raise ValueError("trials must match shape, be >0, and successes <= trials.")
    S = np.concatenate([[0.0], np.cumsum(s)])
    Ntr = np.concatenate([[0.0], np.cumsum(n)])

    g = ncp_prior_from_p0(N, p0) if (p0 is not None) else float(gamma if gamma is not None else 0.0)

    last, cps, opt = _dp_solve(stat_num=S, stat_den=Ntr, fitness_per_block=_fit_bernoulli, gamma=g)

    edges = np.array([0, *cps.tolist(), N], dtype=float)
    block_succ = np.diff(np.concatenate([[0.0], S[cps], [S[-1]]]))
    block_trials = np.diff(np.concatenate([[0.0], Ntr[cps], [Ntr[-1]]]))
    with np.errstate(divide="ignore", invalid="ignore"):
        p_hat = np.where(block_trials > 0, block_succ / block_trials, 0.0)

    return BBResult(edges=edges, block_value=p_hat, change_points=cps)
