# gaussian_diag.py
# MIT License
# (c) 2025

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from typing import NamedTuple

import math
import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.floating]
ArrayI = NDArray[np.integer]

class KMeansResult(NamedTuple):
    centers: ArrayF   # (K, D)
    labels: ArrayI    # (T,)
    inertia: float    # sum of squared distances at convergence
    n_iter: int       # iterations actually run


@dataclass
class GaussianDiagParams:
    """
    Parameters of a K-state diagonal-covariance Gaussian emission model.

    Attributes
    ----------
    mu : (K, D) array
        Mean vectors per state.
    var : (K, D) array
        Variances (diagonal entries) per state. Must be strictly positive.
    """
    mu: ArrayF
    var: ArrayF


# ----------------------------- validation -----------------------------

def _as_2d_float(x: np.ndarray, name: str) -> ArrayF:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D.")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values.")
    return a


def _check_params(params: GaussianDiagParams) -> Tuple[int, int]:
    mu = _as_2d_float(params.mu, "mu")
    var = _as_2d_float(params.var, "var")
    if mu.shape != var.shape:
        raise ValueError("mu and var must have identical shapes (K, D).")
    if np.any(var <= 0.0) or not np.all(np.isfinite(var)):
        raise ValueError("var must be strictly positive and finite.")
    K, D = mu.shape
    return K, D


# ----------------------------- log-likelihood -----------------------------

def gaussian_diag_loglik(
    X: ArrayF,
    params: GaussianDiagParams,
    *,
    allow_nan: bool = False,
    min_var: float = 1e-8,
) -> ArrayF:
    """
    Compute log-likelihoods log p(x_t | state=j) for a diagonal-covariance Gaussian.

    Parameters
    ----------
    X : (T, D) array
        Observations; rows are time steps. If `allow_nan=True`, NaNs indicate missing
        features and are ignored in the likelihood (normalizer adapts to observed dims).
    params : GaussianDiagParams
        Means and variances per state.
    allow_nan : bool, default=False
        If True, treat NaNs in X as missing at random and drop those dimensions per t.
    min_var : float, default=1e-8
        Variance floor for numerical stability (applied elementwise to `params.var`).

    Returns
    -------
    L : (T, K) array
        Log-likelihood matrix suitable for HSMM/HMM routines.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (T, D).")

    K, D = _check_params(params)
    T, D_x = X.shape
    if D_x != D:
        raise ValueError(f"X has D={D_x} but params expect D={D}.")

    mu = params.mu
    var = np.maximum(params.var, float(min_var))
    inv_var = 1.0 / var

    if not allow_nan and np.isnan(X).any():
        raise ValueError("X contains NaN but allow_nan=False.")

    # Precompute per-state log normalizer parts: 0.5 * sum_d log(2π σ^2_d)
    log_norm_per_state = 0.5 * np.sum(np.log(2.0 * math.pi * var), axis=1)  # (K,)

    if not allow_nan:
        # Vectorized broadcast over T x K x D
        diff = X[:, None, :] - mu[None, :, :]                  # (T,K,D)
        quad = np.sum(diff * diff * inv_var[None, :, :], axis=2)  # (T,K)
        L = -0.5 * quad - log_norm_per_state[None, :]          # (T,K)
        return L

    # allow_nan=True: mask missing dims per time, adjust both quadratic and normalizer
    mask = np.isfinite(X)                                      # (T,D)
    X_filled = np.where(mask, X, 0.0)                          # replace NaN by 0 for arithmetic

    # Quadratic term with mask
    diff = X_filled[:, None, :] - mu[None, :, :]               # (T,K,D)
    quad = np.sum(diff * diff * inv_var[None, :, :] * mask[:, None, :], axis=2)  # (T,K)

    # Normalizer term depends on observed dims at t:
    # 0.5 * sum_{d observed at t} log(2π σ^2_{j,d})
    log_var = np.log(2.0 * math.pi * var)                      # (K,D)
    # (T,D) @ (D,K) -> (T,K)
    obs_log_norm = 0.5 * (mask.astype(float) @ log_var.T)

    L = -0.5 * quad - obs_log_norm                             # (T,K)
    return L


# ----------------------------- simple estimators (optional) -----------------------------

def estimate_from_labels(
    X: ArrayF,
    labels: ArrayI,
    K: int,
    *,
    min_var: float = 1e-6,
    allow_nan: bool = False,
) -> GaussianDiagParams:
    """
    MLE of (mu, var) from hard labels.

    Parameters
    ----------
    X : (T, D)
        Observations.
    labels : (T,)
        Integer labels in {0..K-1}.
    K : int
        Number of states.
    min_var : float
        Variance floor for stability.
    allow_nan : bool
        If True, compute means/vars per feature using only observed values at each t.

    Returns
    -------
    GaussianDiagParams
    """
    X = np.asarray(X, dtype=float)
    z = np.asarray(labels, dtype=int)
    if X.ndim != 2 or z.ndim != 1 or X.shape[0] != z.size:
        raise ValueError("Shapes must satisfy X=(T,D), labels=(T,) with matching T.")
    if not allow_nan and np.isnan(X).any():
        raise ValueError("X contains NaN but allow_nan=False.")

    T, D = X.shape
    mu = np.zeros((K, D), dtype=float)
    var = np.zeros((K, D), dtype=float)

    for j in range(K):
        idx = (z == j)
        if not np.any(idx):
            # empty state: fall back to global stats
            mu[j] = np.nanmean(X, axis=0)
            var[j] = np.nanvar(X, axis=0)
            continue
        Xj = X[idx]
        if allow_nan:
            mu[j] = np.nanmean(Xj, axis=0)
            var[j] = np.nanvar(Xj, axis=0)
        else:
            mu[j] = Xj.mean(axis=0)
            var[j] = Xj.var(axis=0)  # MLE (ddof=0)
        var[j] = np.maximum(var[j], min_var)

    return GaussianDiagParams(mu=mu, var=var)


def estimate_from_responsibilities(
    X: ArrayF,
    gamma: ArrayF,   # (T, K)
    *,
    min_var: float = 1e-6,
    allow_nan: bool = False,
) -> GaussianDiagParams:
    """
    Weighted MLE of (mu, var) from soft responsibilities γ_{t,j}.

    Parameters
    ----------
    X : (T, D)
        Observations.
    gamma : (T, K)
        Nonnegative responsibilities; each row may be unnormalized (we normalize per t).
    min_var : float
        Variance floor.

    Returns
    -------
    GaussianDiagParams
    """
    X = np.asarray(X, dtype=float)
    G = np.asarray(gamma, dtype=float)
    if X.ndim != 2 or G.ndim != 2 or X.shape[0] != G.shape[0]:
        raise ValueError("Shapes must satisfy X=(T,D), gamma=(T,K) with matching T.")
    if not allow_nan and np.isnan(X).any():
        raise ValueError("X contains NaN but allow_nan=False.")
    if np.any(G < 0):
        raise ValueError("gamma must be nonnegative.")

    T, D = X.shape
    K = G.shape[1]
    # normalize rows of gamma to sum 1 (safe)
    row_sum = np.clip(G.sum(axis=1, keepdims=True), 1e-12, None)
    Gn = G / row_sum

    mu = np.zeros((K, D), dtype=float)
    var = np.zeros((K, D), dtype=float)

    if allow_nan:
        # treat NaNs as missing: weight only observed entries
        mask = np.isfinite(X).astype(float)  # (T,D)
        X_filled = np.where(mask > 0, X, 0.0)
        for j in range(K):
            w = Gn[:, [j]]                    # (T,1)
            w_sum = (w.T @ mask).reshape(D)   # per-dim effective weight
            w_sum = np.clip(w_sum, 1e-12, None)
            # weighted means per dim
            mu_j_num = (X_filled * mask) * w  # (T,D)
            mu[j] = mu_j_num.sum(axis=0) / w_sum
            # weighted variances per dim
            diff = (X_filled - mu[j]) * mask
            var_num = ((diff * diff) * w).sum(axis=0)
            var[j] = np.maximum(var_num / w_sum, min_var)
    else:
        for j in range(K):
            w = Gn[:, j]                      # (T,)
            ws = np.sum(w) + 1e-12
            # weighted mean
            mu[j] = (w @ X) / ws
            # weighted variance (per dim)
            diff = X - mu[j]
            var[j] = np.maximum((w @ (diff * diff)) / ws, min_var)

    return GaussianDiagParams(mu=mu, var=var)


# --- k-means++ initializer -----------------------------------------------------


def _pairwise_sqdist_normalized(
    X: ArrayF, centers: ArrayF, *, allow_nan: bool
) -> ArrayF:
    """
    Compute normalized squared distances between points and centers.

    If allow_nan:
        dist^2(t,j) = sum_d (x_td - mu_jd)^2 over observed dims / (#observed dims for t)
        If a row has zero observed dims, its distances are set to +inf.

    Returns
    -------
    D2 : (T, K) array of normalized squared distances.
    """
    X = np.asarray(X, dtype=float)
    C = np.asarray(centers, dtype=float)
    T, D = X.shape
    if C.shape[1] != D:
        raise ValueError("centers must have shape (K, D) matching X.")
    if not allow_nan:
        # ||x||^2 + ||mu||^2 - 2 x·mu
        x2 = np.sum(X * X, axis=1, keepdims=True)             # (T,1)
        c2 = np.sum(C * C, axis=1, keepdims=True).T           # (1,K)
        D2 = np.maximum(x2 + c2 - 2.0 * (X @ C.T), 0.0)
        return D2

    # NaN-aware: mask, normalize by #observed dims per row
    mask = np.isfinite(X)                                     # (T,D)
    obs = mask.sum(axis=1, keepdims=True)                     # (T,1)
    # fill NaNs with zeros then subtract centers, masking
    X0 = np.where(mask, X, 0.0)
    # (T,K,D): broadcast subtract centers
    diff = X0[:, None, :] - C[None, :, :]
    diff *= mask[:, None, :]                                  # ignore missing dims
    D2 = np.sum(diff * diff, axis=2)                          # (T,K)
    # normalize (avoid division by 0)
    obs_safe = np.clip(obs, 1, None)
    D2 = D2 / obs_safe
    # rows with zero observed -> +inf so they don't drive assignments
    D2[obs.ravel() == 0] = np.inf
    return D2


def _kmeanspp_seed(
    X: ArrayF, K: int, *, rng: np.random.Generator, allow_nan: bool
) -> ArrayF:
    """
    k-means++ seeding: pick first center uniformly among valid rows,
    then sample next centers ∝ (min distance)^2.
    """
    T, D = X.shape
    # valid rows: at least one observed feature if allow_nan
    if allow_nan:
        valid = np.isfinite(X).any(axis=1)
    else:
        valid = np.ones(T, dtype=bool)
    if not np.any(valid):
        raise ValueError("All rows are fully missing; cannot initialize.")

    idx0 = rng.choice(np.where(valid)[0])
    centers = np.empty((K, D), dtype=float)
    centers[0] = np.where(np.isfinite(X[idx0]), X[idx0], 0.0)

    # initialize distances to first center
    D2 = _pairwise_sqdist_normalized(X, centers[:1], allow_nan=allow_nan).ravel()
    for j in range(1, K):
        # probability ∝ current min squared distance
        probs = np.maximum(D2, 0.0)
        s = probs.sum()
        if not np.isfinite(s) or s <= 0:
            # fall back: pick any valid row farthest from current center set
            cand = int(np.argmax(D2))
        else:
            cand = int(rng.choice(np.arange(T), p=probs / s))
        centers[j] = np.where(np.isfinite(X[cand]), X[cand], 0.0)
        # update min distances
        D2_new = _pairwise_sqdist_normalized(X, centers[: j + 1], allow_nan=allow_nan)
        D2 = np.min(D2_new, axis=1)
    return centers


def _reseed_empty(
    X: ArrayF, labels: ArrayI, centers: ArrayF, *, rng: np.random.Generator, allow_nan: bool
) -> None:
    """
    Detect empty clusters and reseed them at the farthest points from their current nearest center.
    In-place modification of `centers` and `labels` (labels are not reassigned here).
    """
    K = centers.shape[0]
    counts = np.bincount(labels, minlength=K)
    if np.all(counts > 0):
        return
    # distances to nearest center
    D2 = _pairwise_sqdist_normalized(X, centers, allow_nan=allow_nan)
    min_d = np.min(D2, axis=1)
    order = np.argsort(min_d)[::-1]  # farthest first
    ptr = 0
    for j in range(K):
        if counts[j] == 0:
            # pick the next farthest point that is finite
            while ptr < X.shape[0] and not np.all(np.isfinite(D2[order[ptr], :])):
                ptr += 1
            if ptr >= X.shape[0]:
                # fallback to random valid row
                idx = int(rng.integers(0, X.shape[0]))
            else:
                idx = int(order[ptr])
                ptr += 1
            centers[j] = np.where(np.isfinite(X[idx]), X[idx], 0.0)


def kmeanspp_fit(
    X: ArrayF,
    K: int,
    *,
    n_init: int = 4,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: Optional[int] = 123,
    allow_nan: bool = False,
) -> KMeansResult:
    """
    Tiny NumPy-only k-means with k-means++ seeding.

    Parameters
    ----------
    X : (T, D) array
        Data matrix.
    K : int
        Number of clusters/states.
    n_init : int
        Number of random restarts (best inertia kept).
    max_iter : int
        Lloyd iterations per restart.
    tol : float
        Relative tolerance on inertia for early stop.
    seed : Optional[int]
        RNG seed (controls all restarts).
    allow_nan : bool
        If True, distances and updates ignore NaNs, normalized by #observed dims per row.

    Returns
    -------
    KMeansResult(centers, labels, inertia, n_iter)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (T, D).")
    if K < 1 or K > X.shape[0]:
        raise ValueError("K must be in [1, T].")
    rng = np.random.default_rng(seed)

    best = None
    for _ in range(n_init):
        centers = _kmeanspp_seed(X, K, rng=rng, allow_nan=allow_nan)
        inertia_prev = np.inf
        labels = np.zeros(X.shape[0], dtype=int)
        n_iter = 0
        for current_iter in range(1, max_iter + 1):
            n_iter = current_iter
            # Assign
            D2 = _pairwise_sqdist_normalized(X, centers, allow_nan=allow_nan)
            labels = np.argmin(D2, axis=1)
            inertia = float(np.sum(np.min(D2, axis=1)[np.isfinite(np.min(D2, axis=1))]))

            # Update centers (NaN-aware mean)
            for j in range(K):
                mask_j = labels == j
                if not np.any(mask_j):
                    continue
                Xj = X[mask_j]
                if allow_nan:
                    centers[j] = np.nanmean(Xj, axis=0)
                else:
                    centers[j] = Xj.mean(axis=0)

            # Handle empties by reseeding at farthest points
            _reseed_empty(X, labels, centers, rng=rng, allow_nan=allow_nan)

            # Convergence
            if inertia_prev < np.inf:
                if abs(inertia - inertia_prev) <= tol * (1.0 + inertia_prev):
                    break
            inertia_prev = inertia

        res = KMeansResult(centers=centers.copy(), labels=labels.copy(), inertia=inertia, n_iter=n_iter)
        if (best is None) or (res.inertia < best.inertia):
            best = res

    assert best is not None
    return best


def estimate_by_kmeanspp(
    X: ArrayF,
    K: int,
    *,
    n_init: int = 4,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: Optional[int] = 123,
    allow_nan: bool = False,
    min_var: float = 1e-6,
) -> GaussianDiagParams:
    """
    Learn diagonal-Gaussian emission parameters via tiny k-means++.

    Returns
    -------
    GaussianDiagParams(mu, var)
      mu : (K, D) centers
      var: (K, D) per-cluster MLE variances (NaN-aware if allow_nan=True), floored by min_var
    """
    km = kmeanspp_fit(X, K, n_init=n_init, max_iter=max_iter, tol=tol, seed=seed, allow_nan=allow_nan)
    mu = km.centers
    var = np.zeros_like(mu)
    labels = km.labels

    # Per-cluster diagonal variance (MLE), with NaN support and floor
    for j in range(K):
        idx = labels == j
        if not np.any(idx):
            # empty cluster: borrow global variance
            if allow_nan:
                var[j] = np.nanvar(X, axis=0)
            else:
                var[j] = X.var(axis=0)
            continue
        Xj = X[idx]
        if allow_nan:
            var_j = np.nanvar(Xj - mu[j], axis=0)
        else:
            var_j = (Xj - mu[j]).var(axis=0)  # ddof=0 (MLE)
        var[j] = np.maximum(var_j, min_var)

    return GaussianDiagParams(mu=mu, var=var)
