# kcp.py
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
# Kernels and Gram matrices
# ---------------------------------------------------------------------

def _pairwise_sq_dists(X: ArrayF) -> ArrayF:
    """
    Squared Euclidean distances for rows of X in O(n^2) time/memory.

    Returns
    -------
    D2 : (n, n) where D2[i,j] = ||X[i]-X[j]||^2
    """
    X = np.asarray(X, dtype=float)
    s = np.sum(X * X, axis=1, keepdims=True)
    D2 = np.maximum(s + s.T - 2.0 * (X @ X.T), 0.0)
    np.fill_diagonal(D2, 0.0)
    return D2


def gram_linear(X: ArrayF) -> ArrayF:
    """
    Linear kernel Gram matrix: K = X X^T
    """
    X = np.asarray(X, dtype=float)
    return X @ X.T


def gram_rbf(
    X: ArrayF,
    *,
    gamma: float | None = None,
    sigma: float | None = None,
) -> Tuple[ArrayF, float]:
    """
    RBF kernel Gram matrix: K_ij = exp(-gamma * ||x_i - x_j||^2)
    If gamma and sigma are None, uses the median heuristic:
        gamma = 1 / (2 * median(||x_i - x_j||^2))

    Returns
    -------
    K : (n, n) Gram matrix
    gamma : float used
    """
    X = np.asarray(X, dtype=float)
    D2 = _pairwise_sq_dists(X)
    if gamma is None:
        if sigma is not None:
            gamma = 1.0 / (2.0 * (sigma ** 2))
        else:
            # median heuristic
            tri = D2[np.triu_indices(D2.shape[0], k=1)]
            med = float(np.median(tri)) if tri.size else 1.0
            if med <= 0:
                med = 1.0
            gamma = 1.0 / (2.0 * med)
    K = np.exp(-gamma * D2, dtype=float)
    np.fill_diagonal(K, 1.0)
    return K, float(gamma)


# ---------------------------------------------------------------------
# Kernel segment cost via prefix sums
# ---------------------------------------------------------------------

@dataclass
class KernelPrefix:
    """
    Precomputed structures for O(1) kernel segment costs.
    Cost for [i, j) is:
        C(i,j) = sum_{t=i}^{j-1} K_tt - (1/(j-i)) * sum_{p=i}^{j-1} sum_{q=i}^{j-1} K_pq
    This is the within-segment RKHS scatter around the segment mean (constant model in feature space).
    """
    K: ArrayF               # (n, n)
    diag_ps: ArrayF         # (n+1,)
    K_ps2d: ArrayF          # (n, n) inclusive 2D prefix sums of K


def _prefix2d_inclusive(M: ArrayF) -> ArrayF:
    return M.cumsum(axis=0).cumsum(axis=1)


def build_kernel_prefix(K: ArrayF) -> KernelPrefix:
    """
    Build prefix structures from a Gram matrix K.
    """
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    if K.ndim != 2 or K.shape[1] != n:
        raise ValueError("K must be a square (n x n) array.")
    diag_ps = np.concatenate([[0.0], np.cumsum(np.diag(K))])
    K_ps2d = _prefix2d_inclusive(K)
    return KernelPrefix(K=K, diag_ps=diag_ps, K_ps2d=K_ps2d)


def _sum_rect(ps: ArrayF, r0: int, r1: int, c0: int, c1: int) -> float:
    """
    Sum of M[r0:r1, c0:c1] given 2-D inclusive prefix ps on M.
    """
    if r0 >= r1 or c0 >= c1:
        return 0.0
    r1m, c1m = r1 - 1, c1 - 1
    res = ps[r1m, c1m]
    if r0 > 0:
        res -= ps[r0 - 1, c1m]
    if c0 > 0:
        res -= ps[r1m, c0 - 1]
    if r0 > 0 and c0 > 0:
        res += ps[r0 - 1, c0 - 1]
    return float(res)


def kernel_segment_cost(pref: KernelPrefix, i: int, j: int) -> float:
    """
    Kernelized constant-mean cost for a segment [i, j). Requires j > i.

    C(i,j) = sum diag(K[i:j,i:j]) - (1/(j-i)) * sum K[i:j, i:j]
    """
    if not (0 <= i < j <= pref.K.shape[0]):
        return float("inf")
    L = j - i
    diag_sum = float(pref.diag_ps[j] - pref.diag_ps[i])
    block_sum = _sum_rect(pref.K_ps2d, i, j, i, j)
    return diag_sum - (block_sum / max(1, L))


# ---------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------

@dataclass
class KCPResult:
    """
    Result object for kernel change-point detection.
    """
    n: int
    change_points: ArrayI             # sorted CPs in 1..n-1 (right-exclusive)
    labels: ArrayI                    # segment labels 0..K-1 for each index
    total_cost: float                 # objective value
    edges: ArrayI                     # [0, *cps, n]
    costs_per_segment: ArrayF         # cost of each fitted segment


def _labels_from_cps(n: int, cps: ArrayI) -> ArrayI:
    labs = np.empty(n, dtype=int)
    a = 0
    k = 0
    for b in cps.tolist() + [n]:
        labs[a:b] = k
        a = b
        k += 1
    return labs


# ---------------------------------------------------------------------
# Penalized DP (Optimal Partitioning) + PELT pruning
# ---------------------------------------------------------------------

def kcp_penalized(
    pref: KernelPrefix,
    *,
    gamma: float,
    min_size: int = 1,
    method: str = "pelt",   # "pelt" | "op"
    grid_jump: int = 1,     # consider only ends t multiple of grid_jump (approximate speedup)
) -> KCPResult:
    """
    Penalized kernel CPD: minimize sum C(segments) + gamma * m (m = #changepoints).
    - method="pelt": Parallel pruning (expected linear-time under mild conditions).
    - method="op":  Classic O(n^2) optimal partitioning (no pruning).
    - grid_jump>1: evaluate only endpoints t in {grid_jump, 2*grid_jump, ... , n}; always ensure n is included.

    Returns
    -------
    KCPResult
    """
    n = pref.K.shape[0]
    if n < 1:
        return KCPResult(n=0, change_points=np.array([], dtype=np.int64),
                         labels=np.array([], dtype=int), total_cost=0.0,
                         edges=np.array([0], dtype=np.int64), costs_per_segment=np.array([], dtype=float))
    if min_size < 1 or min_size > n:
        raise ValueError("min_size must be in [1, n].")
    if gamma < 0 or not np.isfinite(gamma):
        raise ValueError("gamma must be a non-negative finite number.")
    if method not in {"pelt", "op"}:
        raise ValueError("method must be 'pelt' or 'op'.")

    # Restrict candidate endpoints if grid_jump>1
    grid = np.arange(grid_jump, n, grid_jump, dtype=int).tolist()
    if grid[-1] != n:
        grid = [t for t in grid if t < n] + [n]

    # DP arrays
    F = np.full(n + 1, float("inf"))
    prev = np.full(n + 1, -1, dtype=int)
    seg_cost = np.full(n + 1, float("inf"))

    # base
    F[0] = -gamma
    prev[0] = -1

    if method == "op":
        # O(n^2) optimal partitioning
        for t in grid:
            # admissible starts i
            i_min = max(0, t - (10 ** 9))    # no max len; placeholder
            i_max = t - min_size
            if i_max < 0:
                continue
            idx = np.arange(i_min, i_max + 1, dtype=int)
            costs = np.fromiter((kernel_segment_cost(pref, i, t) for i in idx), count=idx.size, dtype=float)
            vals = F[idx] + costs + gamma
            k = int(np.argmin(vals))
            F[t] = float(vals[k])
            prev[t] = int(idx[k])
            seg_cost[t] = float(costs[k])

    else:
        # PELT pruning
        R: List[int] = [0]  # candidate last-CP positions
        for t in grid:
            # restrict candidates by min_size
            Rt = [i for i in R if t - i >= min_size]
            if not Rt:
                continue
            costs = np.fromiter((kernel_segment_cost(pref, i, t) for i in Rt), count=len(Rt), dtype=float)
            vals = F[np.array(Rt)] + costs + gamma
            k = int(np.argmin(vals))
            F[t] = float(vals[k])
            prev[t] = int(Rt[k])
            seg_cost[t] = float(costs[k])

            # update candidate set with pruning:
            # keep i in Rt_plus if F[i] + C(i,t) <= F[t]
            Rt_plus = Rt + [t - min_size]  # newly admissible start for next step
            R = []
            for i in Rt_plus:
                if F[i] + kernel_segment_cost(pref, i, t) <= F[t] + 1e-12:
                    R.append(i)

    # Backtrack
    cps: List[int] = []
    costs_list: List[float] = []
    t = n
    if F[t] == float("inf"):
        # fallback: force a single segment
        cps = []
        costs_list = [kernel_segment_cost(pref, 0, n)]
        edges = np.array([0, n], dtype=np.int64)
        labels = np.zeros(n, dtype=int)
        return KCPResult(n=n, change_points=np.array([], dtype=np.int64),
                         labels=labels, total_cost=float(costs_list[0]),
                         edges=edges, costs_per_segment=np.array(costs_list))
    while t > 0:
        i = int(prev[t])
        cps.append(t)
        costs_list.append(kernel_segment_cost(pref, i, t))
        t = i
    cps = list(reversed(cps[:-1]))  # drop the terminal n
    edges = np.array([0, *cps, n], dtype=np.int64)
    labels = _labels_from_cps(n, np.asarray(cps, dtype=np.int64))
    return KCPResult(n=n, change_points=np.asarray(cps, dtype=np.int64),
                     labels=labels, total_cost=float(F[n]), edges=edges,
                     costs_per_segment=np.asarray(costs_list[::-1], dtype=float))


# ---------------------------------------------------------------------
# Fixed-m segmentation (segment neighborhood DP)
# ---------------------------------------------------------------------

@dataclass
class KCPSNResult:
    """
    Segment-neighborhood dynamic programming for fixed number of segments m.
    """
    n: int
    m: int
    edges: ArrayI
    change_points: ArrayI
    labels: ArrayI
    total_cost: float


def kcp_fixed_m(
    pref: KernelPrefix,
    *,
    m: int,
    min_size: int = 1,
    grid_jump: int = 1,
) -> KCPSNResult:
    """
    Exact dynamic programming for the best segmentation into exactly m segments.
    Complexity: O(n^2 m). Use moderate m (<= ~20) or increase grid_jump.

    Returns
    -------
    KCPSNResult
    """
    n = pref.K.shape[0]
    if m < 1 or m > n:
        raise ValueError("m must be in [1, n].")
    if min_size < 1 or min_size > n:
        raise ValueError("min_size must be in [1, n].")

    # grid of candidate endpoints
    grid = np.arange(grid_jump, n, grid_jump, dtype=int).tolist()
    if not grid or grid[-1] != n:
        grid = [t for t in grid if t < n] + [n]
    # We need dynamic tables over endpoint indices in 0..n
    T = len(grid)
    endpoints = np.array([0] + grid, dtype=int)  # allow 0
    # DP tables
    F = np.full((m + 1, T + 1), float("inf"))
    P = np.full((m + 1, T + 1), -1, dtype=int)
    F[0, 0] = 0.0  # 0 segments up to 0 cost 0

    for k in range(1, m + 1):
        for j in range(1, T + 1):
            t = endpoints[j]
            # candidates i are endpoints[:j] with min_size
            best_val = float("inf")
            best_i = -1
            for i_idx in range(0, j):
                i = endpoints[i_idx]
                if t - i < min_size:
                    continue
                c = kernel_segment_cost(pref, i, t)
                val = F[k - 1, i_idx] + c
                if val < best_val:
                    best_val = val
                    best_i = i_idx
            F[k, j] = best_val
            P[k, j] = best_i

    # backtrack best at k=m, j=T
    cps: List[int] = []
    cur_k, cur_j = m, T
    while cur_k > 0:
        i_idx = int(P[cur_k, cur_j])
        t = int(endpoints[cur_j])
        i = int(endpoints[i_idx])
        cps.append(t)
        cur_k -= 1
        cur_j = i_idx
    cps = list(reversed(cps[:-1]))  # drop terminal n
    edges = np.array([0, *cps, n], dtype=int)
    labels = _labels_from_cps(n, np.asarray(cps, dtype=int))
    return KCPSNResult(n=n, m=m, edges=edges, change_points=np.asarray(cps, dtype=int),
                       labels=labels, total_cost=float(F[m, T]))


# ---------------------------------------------------------------------
# BIC-style model selection
# ---------------------------------------------------------------------

@dataclass
class KCPModelSel:
    """
    Model selection over m using a BIC-style linear penalty: cost + beta * m * log(n)
    """
    m_star: int
    edges: ArrayI
    change_points: ArrayI
    labels: ArrayI
    costs_m: ArrayF
    penalized_m: ArrayF
    beta: float


def kcp_select_bic(
    pref: KernelPrefix,
    *,
    m_max: int,
    beta: float = 1.0,
    min_size: int = 1,
    grid_jump: int = 1,
) -> KCPModelSel:
    """
    Evaluate the best fixed-m segmentation for m=1..m_max, then pick
    m* = argmin_m { cost_m + beta * m * log(n) }.

    Notes
    -----
    - 'beta' is a user scale since the RKHS 'dimension' is implicit.
      Start with beta in [0.5, 2] and adjust by inspection; or compare to a
      penalized fit with gamma ~= beta * log(n).
    """
    n = pref.K.shape[0]
    costs = np.empty(m_max, dtype=float)
    edges_list: List[ArrayI] = []
    cps_list: List[ArrayI] = []
    labels_list: List[ArrayI] = []

    for m in range(1, m_max + 1):
        res = kcp_fixed_m(pref, m=m, min_size=min_size, grid_jump=grid_jump)
        costs[m - 1] = res.total_cost
        edges_list.append(res.edges)
        cps_list.append(res.change_points)
        labels_list.append(res.labels)

    pen = costs + beta * np.log(max(2, n)) * np.arange(1, m_max + 1)
    k = int(np.argmin(pen))
    return KCPModelSel(
        m_star=k + 1,
        edges=edges_list[k],
        change_points=cps_list[k],
        labels=labels_list[k],
        costs_m=costs,
        penalized_m=pen,
        beta=beta,
    )
