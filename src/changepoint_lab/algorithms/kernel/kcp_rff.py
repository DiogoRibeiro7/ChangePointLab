# kcp_rff.py
# MIT License
# (c) 2025

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import math
import numpy as np
from numpy.typing import NDArray

# Scientific traceability:
# - Rahimi and Recht (2007), Random Features for Large-Scale Kernel Machines.
# - Registry entry: docs/science/method_registry.yml, method id "rff_kernel_cpd".

ArrayF = NDArray[np.floating]
ArrayI = NDArray[np.integer]


# --------------------------- RFF configuration ---------------------------

@dataclass
class RFFConfig:
    """
    Random Fourier Features for the RBF kernel k(x,y) = exp(-gamma * ||x - y||^2).

    Parameters
    ----------
    n_features : int
        Output feature dimension D for the RFF embedding.
    gamma : Optional[float]
        RBF parameter (if None, we estimate from data with a robust median-heuristic on a subsample).
    sigma : Optional[float]
        Alternative to gamma. If provided, gamma = 1 / (2 * sigma^2).
    subsample_for_bandwidth : int
        Max number of points used for bandwidth estimation (median heuristic). Keeps it sub-quadratic.
    seed : Optional[int]
        RNG seed for reproducibility.
    """
    n_features: int = 512
    gamma: Optional[float] = None
    sigma: Optional[float] = None
    subsample_for_bandwidth: int = 2000
    seed: Optional[int] = 123


# --------------------------- RFF embedding ---------------------------

def _estimate_gamma_median_heuristic(
    X: ArrayF, max_samples: int, rng: np.random.Generator
) -> float:
    """
    Subsample-based median heuristic:
        gamma = 1 / (2 * median(||x_i - x_j||^2))
    Uses up to max_samples points (O(m^2) with m <= max_samples << n).
    """
    n = X.shape[0]
    m = min(n, int(max_samples))
    if m < 2:
        return 1.0  # fallback
    idx = rng.choice(n, size=m, replace=False)
    Xm = X[idx]
    s = np.sum(Xm * Xm, axis=1, keepdims=True)
    D2 = np.maximum(s + s.T - 2.0 * (Xm @ Xm.T), 0.0)
    tri = D2[np.triu_indices(m, k=1)]
    med = float(np.median(tri)) if tri.size else 1.0
    if med <= 0.0:
        med = 1.0
    return 1.0 / (2.0 * med)


@dataclass
class RFFMap:
    """
    Holder for RFF parameters and the embedded matrix Z.
    """
    Z: ArrayF          # (n, D) feature matrix
    gamma: float       # gamma actually used
    W: ArrayF          # (d, D) projection matrix
    b: ArrayF          # (D,) phase vector


def rbf_rff_map(
    X: ArrayF | Sequence[float],
    cfg: RFFConfig = RFFConfig(),
    sigma: float | None = None,
    gamma: float | None = None,
) -> RFFMap:
    """
    Build the Random Fourier Features embedding for an RBF kernel.

    Returns
    -------
    RFFMap with Z (n,D), gamma, and the sampled (W,b).
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    n, d = X_arr.shape
    if cfg.n_features < 1:
        raise ValueError("n_features must be >= 1.")
    rng = np.random.default_rng(cfg.seed)

    # Resolve gamma
    if gamma is not None:
        gamma = float(gamma)
    elif sigma is not None:
        gamma = 1.0 / (2.0 * float(sigma) ** 2)
    elif cfg.gamma is not None:
        gamma = float(cfg.gamma)
    elif cfg.sigma is not None:
        gamma = 1.0 / (2.0 * float(cfg.sigma) ** 2)
    else:
        gamma = _estimate_gamma_median_heuristic(
            X_arr, max_samples=cfg.subsample_for_bandwidth, rng=rng
        )

    # Sample RFF parameters
    # For k(x,y) = exp(-gamma ||x-y||^2), omega ~ N(0, 2*gamma I)
    W = rng.normal(loc=0.0, scale=math.sqrt(2.0 * gamma), size=(d, cfg.n_features))
    b = rng.uniform(low=0.0, high=2.0 * math.pi, size=(cfg.n_features,))

    # Embed
    proj = X_arr @ W + b  # (n,D)
    Z = math.sqrt(2.0 / cfg.n_features) * np.cos(proj, dtype=float)
    return RFFMap(Z=Z, gamma=float(gamma), W=W, b=b)


# --------------------------- Feature prefix sums ---------------------------

@dataclass
class FeaturePrefix:
    """
    Prefix sums for O(1) segment cost in feature space.

    We store:
        S[t] = sum_{u < t} Z[u]        (vector, shape (D,))
        Q[t] = sum_{u < t} ||Z[u]||^2  (scalar)
    for t = 0..n.
    """
    S: ArrayF    # (n+1, D)
    Q: ArrayF    # (n+1,)


def build_feature_prefix(Z: ArrayF) -> FeaturePrefix:
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError("Z must be 2-D (n, D).")
    n, D = Z.shape
    S = np.empty((n + 1, D), dtype=float)
    S[0] = 0.0
    np.cumsum(Z, axis=0, out=S[1:])
    Q = np.empty(n + 1, dtype=float)
    Q[0] = 0.0
    np.cumsum(np.sum(Z * Z, axis=1), out=Q[1:])
    return FeaturePrefix(S=S, Q=Q)


def feature_segment_cost(pref: FeaturePrefix, i: int, j: int) -> float:
    """
    Cost of constant-mean fit on [i, j) in the RFF space:
        C = (sum ||z_t||^2) - (1/L) * ||sum z_t||^2
    """
    if not (0 <= i < j <= pref.S.shape[0] - 1):
        return float("inf")
    L = j - i
    Qseg = float(pref.Q[j] - pref.Q[i])
    Sseg = pref.S[j] - pref.S[i]
    return Qseg - float(Sseg @ Sseg) / max(1, L)


# --------------------------- Results container ---------------------------

@dataclass
class RFFKCPResult:
    """
    Result for RFF-based kernel CPD.
    """
    n: int
    change_points: ArrayI
    labels: ArrayI
    total_cost: float
    edges: ArrayI
    costs_per_segment: ArrayF
    rff_gamma: float
    n_features: int


def _labels_from_cps(n: int, cps: ArrayI) -> ArrayI:
    y = np.empty(n, dtype=int)
    a = 0
    k = 0
    for b in cps.tolist() + [n]:
        y[a:b] = k
        a = b
        k += 1
    return y


# --------------------------- Penalized DP (PELT/OP) ---------------------------

def rff_kcp_penalized(
    pref: FeaturePrefix,
    *,
    gamma_pen: float,
    min_size: int = 1,
    method: str = "pelt",   # "pelt" or "op"
    grid_jump: int = 1,
) -> RFFKCPResult:
    """
    Penalized optimal partitioning (constant mean in RFF space):
        minimize  sum C(seg) + gamma_pen * m.

    - method="pelt": expected linear time with pruning.
    - method="op":   classic O(n^2).
    - grid_jump>1:   evaluate only endpoints t in {k*grid_jump}; always includes n.
    """
    n = pref.S.shape[0] - 1
    if n < 1:
        return RFFKCPResult(n=0, change_points=np.array([], dtype=int),
                            labels=np.array([], dtype=int), total_cost=0.0,
                            edges=np.array([0], dtype=int),
                            costs_per_segment=np.array([], dtype=float),
                            rff_gamma=float("nan"), n_features=pref.S.shape[1])
    if min_size < 1 or min_size > n:
        raise ValueError("min_size must be in [1, n].")
    if gamma_pen < 0 or not np.isfinite(gamma_pen):
        raise ValueError("gamma_pen must be non-negative and finite.")
    if method not in {"pelt", "op"}:
        raise ValueError("method must be 'pelt' or 'op'.")

    # endpoint grid
    grid = np.arange(grid_jump, n, grid_jump, dtype=int).tolist()
    if not grid or grid[-1] != n:
        grid = [t for t in grid if t < n] + [n]

    F = np.full(n + 1, float("inf"))
    prev = np.full(n + 1, -1, dtype=int)
    seg_c = np.full(n + 1, float("inf"))
    F[0] = -gamma_pen

    if method == "op":
        for t in grid:
            i_max = t - min_size
            if i_max < 0:
                continue
            idx = np.arange(0, i_max + 1, dtype=int)
            costs = np.fromiter((feature_segment_cost(pref, i, t) for i in idx), count=idx.size, dtype=float)
            vals = F[idx] + costs + gamma_pen
            k = int(np.argmin(vals))
            F[t] = float(vals[k])
            prev[t] = int(idx[k])
            seg_c[t] = float(costs[k])
    else:
        R: List[int] = [0]
        for t in grid:
            Rt = [i for i in R if (t - i) >= min_size]
            if not Rt:
                continue
            costs = np.fromiter((feature_segment_cost(pref, i, t) for i in Rt), count=len(Rt), dtype=float)
            vals = F[np.array(Rt)] + costs + gamma_pen
            k = int(np.argmin(vals))
            F[t] = float(vals[k])
            prev[t] = int(Rt[k])
            seg_c[t] = float(costs[k])

            Rt_plus = Rt + [t - min_size]
            R = []
            for i in Rt_plus:
                if F[i] + feature_segment_cost(pref, i, t) <= F[t] + 1e-12:
                    R.append(i)

    # backtrack
    cps: List[int] = []
    costs_list: List[float] = []
    t = n
    if not np.isfinite(F[t]):
        # one segment fallback
        one = feature_segment_cost(pref, 0, n)
        return RFFKCPResult(n=n, change_points=np.array([], dtype=int),
                            labels=np.zeros(n, dtype=int), total_cost=float(one),
                            edges=np.array([0, n], dtype=int), costs_per_segment=np.array([one], dtype=float),
                            rff_gamma=float("nan"), n_features=pref.S.shape[1])
    while t > 0:
        i = int(prev[t])
        cps.append(t)
        costs_list.append(feature_segment_cost(pref, i, t))
        t = i
    cps = list(reversed(cps[:-1]))
    edges = np.array([0, *cps, n], dtype=int)
    labels = _labels_from_cps(n, np.asarray(cps, dtype=int))
    return RFFKCPResult(n=n, change_points=np.asarray(cps, dtype=int), labels=labels,
                        total_cost=float(F[n]), edges=edges,
                        costs_per_segment=np.asarray(costs_list[::-1], dtype=float),
                        rff_gamma=float("nan"), n_features=pref.S.shape[1])


# --------------------------- Fixed-m (segment neighborhood) ---------------------------

@dataclass
class RFFKCPSNResult:
    n: int
    m: int
    edges: ArrayI
    change_points: ArrayI
    labels: ArrayI
    total_cost: float
    n_features: int


def rff_kcp_fixed_m(
    pref: FeaturePrefix,
    *,
    m: int,
    min_size: int = 1,
    grid_jump: int = 1,
) -> RFFKCPSNResult:
    """
    Exact DP for exactly m segments in the RFF space. Complexity O(n^2 m)
    (use moderate m or increase grid_jump for speed).
    """
    n = pref.S.shape[0] - 1
    if m < 1 or m > n:
        raise ValueError("m must be in [1, n].")
    if min_size < 1 or min_size > n:
        raise ValueError("min_size must be in [1, n].")

    grid = np.arange(grid_jump, n, grid_jump, dtype=int).tolist()
    if not grid or grid[-1] != n:
        grid = [t for t in grid if t < n] + [n]
    endpoints = np.array([0] + grid, dtype=int)
    T = endpoints.size - 1

    F = np.full((m + 1, T + 1), float("inf"))
    P = np.full((m + 1, T + 1), -1, dtype=int)
    F[0, 0] = 0.0

    for k in range(1, m + 1):
        for j in range(1, T + 1):
            t = endpoints[j]
            best = float("inf")
            best_i = -1
            for i_idx in range(0, j):
                i = endpoints[i_idx]
                if t - i < min_size:
                    continue
                c = feature_segment_cost(pref, i, t)
                val = F[k - 1, i_idx] + c
                if val < best:
                    best = val
                    best_i = i_idx
            F[k, j] = best
            P[k, j] = best_i

    cps: List[int] = []
    k, j = m, T
    while k > 0:
        i_idx = int(P[k, j])
        t = int(endpoints[j])
        i = int(endpoints[i_idx])
        cps.append(t)
        k -= 1
        j = i_idx
    cps = list(reversed(cps[:-1]))
    edges = np.array([0, *cps, n], dtype=int)
    labels = _labels_from_cps(n, np.asarray(cps, dtype=int))
    return RFFKCPSNResult(n=n, m=m, edges=edges, change_points=np.asarray(cps, dtype=int),
                          labels=labels, total_cost=float(F[m, T]), n_features=pref.S.shape[1])
