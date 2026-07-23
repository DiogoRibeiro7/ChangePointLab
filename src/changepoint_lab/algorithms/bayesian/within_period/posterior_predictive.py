# posterior_predictive.py
# MIT License

from __future__ import annotations
from typing import Dict, Sequence, Tuple, List

import numpy as np
from numpy.typing import NDArray

Tau = Tuple[int, ...]


def _mod_distance(a: int, b: int, N: int) -> int:
    d = (b - a) % N
    return d


def _segment_lengths(tau: Tau, N: int) -> List[int]:
    if len(tau) == 0:
        return [N]
    lens: List[int] = []
    prev = tau[-1]
    for t in tau:
        d = _mod_distance(prev, t, N)
        lens.append(N if d == 0 else d)
        prev = t
    return lens


def _sample_phi_for_tau(tau: Tau, s_list: Sequence[int], n_list: Sequence[int], rng: np.random.Generator) -> NDArray[np.floating]:
    """
    Draw segment probabilities phi_i ~ Beta(1+s_i, 1+(n_i-s_i)) and expand them to an N-grid per tau.
    """
    m = 1 if len(tau) == 0 else len(tau) + 1
    alphas = np.array([1 + s for s in s_list], dtype=float)
    betas  = np.array([1 + (n - s) for s, n in zip(s_list, n_list)], dtype=float)
    phi_seg = rng.beta(alphas, betas, size=m)
    return phi_seg


def _grid_from_phi_seg(tau: Tau, phi_seg: NDArray[np.floating], N: int) -> NDArray[np.floating]:
    grid = np.empty(N, dtype=float)
    if len(tau) == 0:
        grid[:] = phi_seg[0]
        return grid
    prev = tau[-1]
    for idx, cp in enumerate(tau):
        length = _mod_distance(prev, cp, N)
        length = N if length == 0 else length
        a = (prev + 1) % N
        for k in range(length):
            grid[(a + k) % N] = phi_seg[idx]
        prev = cp
    return grid


def posterior_predictive_daily_counts(
    *,
    samples_tau: Sequence[Tau],
    seg_stats_for_sample: Sequence[Tuple[Sequence[int], Sequence[int]]],
    N: int,
    days: int,
    replications: int = 200,
    seed: int | None = 123,
) -> Dict[str, NDArray]:
    """
    Simulate posterior predictive 'days' of activity counts per time-of-day bin.

    Parameters
    ----------
    samples_tau : sequence of Tau
        Kept RJMCMC samples.
    seg_stats_for_sample : sequence of (s_list, n_list)
        Precomputed per-sample segment stats (from model._segment_stats(tau)).
        Must match 1-to-1 with samples_tau.
    N : int
        Period length (bins in a day).
    days : int
        Number of synthetic days per replication.
    replications : int, default=200
        How many posterior draws to simulate.
    seed : Optional[int], default=123
        RNG seed.

    Returns
    -------
    dict with keys:
        'sim_mean' : (N,) posterior predictive mean count per bin (averaged across replications & days)
        'sim_lower': (N,) 2.5% quantile (approx) across replications
        'sim_upper': (N,) 97.5% quantile (approx) across replications
    """
    rng = np.random.default_rng(seed)
    R = min(replications, len(samples_tau))
    if R == 0:
        raise ValueError("No samples provided.")
    means = np.zeros((R, N), dtype=float)

    # Randomly subsample R posterior draws (without replacement if possible)
    idx = rng.choice(len(samples_tau), size=R, replace=(R > len(samples_tau)))
    for j, k in enumerate(idx):
        tau = samples_tau[k]
        s_list, n_list = seg_stats_for_sample[k]
        phi_seg = _sample_phi_for_tau(tau, s_list, n_list, rng)
        phi_grid = _grid_from_phi_seg(tau, phi_seg, N)
        # Simulate 'days' Bernoulli trials per bin and average
        draws = rng.binomial(1, phi_grid, size=(days, N)).mean(axis=0)
        means[j, :] = draws

    sim_mean  = means.mean(axis=0)
    sim_lower = np.quantile(means, 0.025, axis=0)
    sim_upper = np.quantile(means, 0.975, axis=0)
    return {"sim_mean": sim_mean, "sim_lower": sim_lower, "sim_upper": sim_upper}



# # Build seg stats for each kept sample once
# seg_stats = [model._segment_stats(tau) for tau in result.samples_tau]  # uses model internals you already have

# from posterior_predictive import posterior_predictive_daily_counts
# ppc = posterior_predictive_daily_counts(
#     samples_tau=result.samples_tau,
#     seg_stats_for_sample=seg_stats,
#     N=prior.N,
#     days=30,
#     replications=200,
# )

# Compare ppc['sim_mean'] to your empirical per-bin mean (observed)
