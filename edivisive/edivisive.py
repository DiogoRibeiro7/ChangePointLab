# edivisive.py
# MIT License
# (c) 2025

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.floating]
ArrayI = NDArray[np.integer]


# ------------------------------- Result container -------------------------------

@dataclass
class EDivisiveSplit:
    """A single accepted changepoint found within a tested segment."""
    index: int             # global index of changepoint (split point, right-exclusive)
    seg_start: int         # segment start (inclusive) where it was found
    seg_end: int           # segment end (exclusive) where it was found
    statistic: float       # observed test statistic at the accepted split
    pvalue: float          # permutation p-value for this split


@dataclass
class EDivisiveResult:
    """Complete E-Divisive result for one sequence."""
    n: int
    alpha: float
    min_size: int
    splits: List[EDivisiveSplit]     # in the order they were accepted
    change_points: ArrayI            # sorted unique changepoint indices
    labels: ArrayI                   # segment labels 0..K-1 for each time index

# ------------------------------- Resamplers -------------------------------

def _choose_block_size(m: int, user_b: Optional[int]) -> int:
    """
    Pick a block size for block-resampling inside a segment of length m.

    Heuristic default: ceil(1.5 * m^(1/3)), clamped to [2, m].
    This preserves short-range dependence but still mixes blocks.
    """
    if user_b is None:
        b = int(np.ceil(1.5 * (m ** (1.0 / 3.0))))
    else:
        b = int(user_b)
    return max(2, min(b, m))


def _resample_iid_permutation(m: int, rng: np.random.Generator) -> ArrayI:
    """Full i.i.d. permutation of indices 0..m-1."""
    return rng.permutation(m).astype(int, copy=False)


def _resample_block_permutation(m: int, b: int, rng: np.random.Generator) -> ArrayI:
    """
    Shuffle non-overlapping contiguous blocks of length b (last block may be shorter).
    Produces a true permutation (no repeats, no omissions).
    """
    starts = list(range(0, m, b))
    order = rng.permutation(len(starts))
    idx_list: list[int] = []
    for k in order:
        a = starts[k]
        b_end = min(a + b, m)
        idx_list.extend(range(a, b_end))
    return np.asarray(idx_list, dtype=int)


def _resample_circular_block_bootstrap(m: int, b: int, rng: np.random.Generator) -> ArrayI:
    """
    Circular moving-block bootstrap (CBB) with replacement:
    repeatedly draw a random start s and take indices s..s+b-1 modulo m,
    concatenating until length >= m; then truncate to m.
    """
    out = np.empty(m, dtype=int)
    filled = 0
    while filled < m:
        s = int(rng.integers(0, m))
        block = (s + np.arange(b)) % m
        k = min(b, m - filled)
        out[filled:filled + k] = block[:k]
        filled += k
    return out

# ------------------------------- Distance helpers -------------------------------

def _pairwise_energy_dist_alpha(X: ArrayF, alpha: float) -> ArrayF:
    """
    Compute pairwise Euclidean distances raised to the power alpha:
        D[i,j] = ||X[i]-X[j]||_2 ** alpha
    Efficient via the squared distance identity; alpha in (0, 2].
    """
    if not (0.0 < alpha <= 2.0):
        raise ValueError("alpha must be in (0, 2].")
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    # squared distances via ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    s = np.sum(X * X, axis=1, keepdims=True)
    dist2 = np.maximum(s + s.T - 2.0 * (X @ X.T), 0.0)
    # distance^alpha = (distance^2)^(alpha/2)
    if alpha == 2.0:
        D = dist2  # (||x-y||^2)^{1} = ||x-y||^2
    else:
        D = np.power(dist2, alpha / 2.0, where=(dist2 > 0.0), out=np.zeros_like(dist2))
    np.fill_diagonal(D, 0.0)
    return D


def _prefix2d(M: ArrayF) -> ArrayF:
    """
    2-D inclusive prefix sums: PS[i,j] = sum_{r<=i, c<=j} M[r,c].
    Allows O(1) rectangular sum queries.
    """
    PS = M.cumsum(axis=0).cumsum(axis=1)
    return PS


def _sum_rect(ps: ArrayF, r0: int, r1: int, c0: int, c1: int) -> float:
    """
    Sum of M[r0:r1, c0:c1] (half-open) given 2-D inclusive prefix ps on M.
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


# ------------------------------- Core statistic -------------------------------

def _energy_stat_scan_from_ps(ps: ArrayF) -> Tuple[ArrayF, ArrayF, ArrayF]:
    """
    For a segment represented by its distance matrix D (m x m) and its 2-D prefix sums ps,
    compute for each split a in {1..m-1}:

      E(a) = (nL*nR)/(nL+nR) * [ 2/(nL nR) * S_cross
                                 - 1/nL^2 * S_LL
                                 - 1/nR^2 * S_RR ]

    where:
      nL = a, nR = m-a
      S_LL = sum D[i,j] for i,j in [0..a-1]
      S_RR = sum D[i,j] for i,j in [a..m-1]
      S_cross = sum D[i,j] for i in [0..a-1], j in [a..m-1]

    Returns arrays of length m-1: (E, S_cross, S_LL_plus_S_RR)
    """
    m = ps.shape[0]
    E = np.empty(m - 1, dtype=float)
    S_cross_arr = np.empty(m - 1, dtype=float)
    S_within_arr = np.empty(m - 1, dtype=float)

    S_total = _sum_rect(ps, 0, m, 0, m)  # full sum

    for a in range(1, m):  # split point
        nL, nR = a, m - a
        S_LL = _sum_rect(ps, 0, a, 0, a)
        S_RR = _sum_rect(ps, a, m, a, m)
        # cross only top-left block vs right block (pairs counted once)
        S_cross = _sum_rect(ps, 0, a, a, m)

        E[a - 1] = (nL * nR) / (nL + nR) * (
            (2.0 * S_cross) / (nL * nR)
            - S_LL / (nL * nL)
            - S_RR / (nR * nR)
        )
        S_cross_arr[a - 1] = S_cross
        S_within_arr[a - 1] = S_LL + S_RR

    return E, S_cross_arr, S_within_arr


def _best_split_statistic(D: ArrayF, min_size: int) -> Tuple[int, float, ArrayF]:
    """
    Given a segment distance matrix D (m x m), return:
      - best split index a* in {min_size .. m-min_size}
      - its statistic E*
      - the full profile E[a] (with NaN outside admissible splits)
    """
    m = D.shape[0]
    ps = _prefix2d(D)
    E, _, _ = _energy_stat_scan_from_ps(ps)

    # Mask out invalid splits that violate min_size
    mask = np.ones_like(E, dtype=bool)
    mask[: max(0, min_size - 1)] = False
    mask[m - min_size - 1 :] = False

    E_masked = np.where(mask, E, -np.inf)
    a_star_rel = int(np.argmax(E_masked)) + 1  # +1 to convert to split index
    E_star = float(E_masked[a_star_rel - 1])
    # Put NaN for plotting outside valid region
    E_plot = np.where(mask, E, np.nan)
    return a_star_rel, E_star, E_plot


def _permutation_max_stat(
    D: ArrayF,
    *,
    R: int,
    min_size: int,
    rng: np.random.Generator,
) -> float:
    """
    Permutation test under null: randomly permute the order within the segment,
    recompute the maximal statistic across splits, and return that maximum.
    """
    m = D.shape[0]
    best = -np.inf
    for _ in range(R):
        idx = rng.permutation(m)
        Dp = D[np.ix_(idx, idx)]
        a_rel, E_star, _ = _best_split_statistic(Dp, min_size=min_size)
        if E_star > best:
            best = E_star
    return float(best)


# ------------------------------- Public API -------------------------------

def edivisive(
    X: ArrayF | Sequence[float],
    *,
    alpha: float = 1.0,
    min_size: int = 20,
    R: int = 499,
    significance: float = 0.05,
    max_cps: Optional[int] = None,
    seed: Optional[int] = 123,
    progress: bool = False,
    # --- new options ---
    resample: str = "iid",                 # "iid" | "block-permutation" | "circular-block-bootstrap"
    block_size: Optional[int] = None,      # block length for block-based resampling
) -> EDivisiveResult:
    """
    E-Divisive multiple changepoint detection using energy statistics.

    Parameters
    ----------
    X : array-like (n,d) or (n,)
        Observations (multivariate allowed).
    alpha : float in (0, 2], default=1.0
        Distance exponent.
    min_size : int, default=20
        Minimum size on both sides of a split.
    R : int, default=499
        Number of resamples for the null distribution on each tested segment.
    significance : float, default=0.05
        p-value threshold.
    max_cps : Optional[int]
        Maximum number of changepoints to accept globally.
    seed : Optional[int]
        RNG seed.
    progress : bool
        Print short progress lines.
    resample : {"iid", "block-permutation", "circular-block-bootstrap"}, default="iid"
        Resampling scheme for the null:
          - "iid": full permutation (independent observations).
          - "block-permutation": shuffle contiguous, non-overlapping blocks of length `block_size`.
          - "circular-block-bootstrap": moving blocks with replacement (CBB) of length `block_size`.
        Block methods preserve short-range dependence inside blocks.
    block_size : Optional[int]
        Block length. If None, an automatic rule is used per tested segment:
        ceil(1.5 * m^(1/3)), clamped to [2, m], where m is the segment length.
    """
    # ------------- validate & shape -------------
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    n, d = X_arr.shape
    if n < 2 * min_size:
        raise ValueError(f"Need at least 2*min_size observations; got n={n}, min_size={min_size}.")
    if not (0.0 < alpha <= 2.0):
        raise ValueError("alpha must be in (0, 2].")
    if R < 1:
        raise ValueError("R must be >= 1.")
    if not (0.0 < significance < 1.0):
        raise ValueError("significance must be in (0,1).")
    if resample not in {"iid", "block-permutation", "circular-block-bootstrap"}:
        raise ValueError("resample must be one of {'iid','block-permutation','circular-block-bootstrap'}.")

    rng = np.random.default_rng(seed)

    # Worklist for divisive recursion
    segments: List[Tuple[int, int]] = [(0, n)]
    accepted: List[EDivisiveSplit] = []
    total_cps = 0

    while segments:
        s, e = segments.pop(0)
        m = e - s
        if m < 2 * min_size:
            continue

        # Distance matrix on the segment
        D = _pairwise_energy_dist_alpha(X_arr[s:e], alpha=alpha)

        # Observed best split
        a_rel, E_star, _ = _best_split_statistic(D, min_size=min_size)
        if not np.isfinite(E_star) or E_star <= 0.0:
            continue

        # Build resampler for this segment
        if resample == "iid":
            def _draw_idx() -> ArrayI:
                return _resample_iid_permutation(m, rng)
        elif resample == "block-permutation":
            b = _choose_block_size(m, block_size)
            def _draw_idx(b=b) -> ArrayI:
                return _resample_block_permutation(m, b, rng)
        else:  # "circular-block-bootstrap"
            b = _choose_block_size(m, block_size)
            def _draw_idx(b=b) -> ArrayI:
                return _resample_circular_block_bootstrap(m, b, rng)

        # Null distribution of the max statistic under chosen resampling
        max_null = np.empty(R, dtype=float)
        for r in range(R):
            idx = _draw_idx()
            Dp = D[np.ix_(idx, idx)]
            _, Enull, _ = _best_split_statistic(Dp, min_size=min_size)
            max_null[r] = Enull

        # Unbiased permutation p-value
        pval = (1.0 + np.sum(max_null >= E_star)) / (R + 1.0)

        if progress:
            meth = resample if resample == "iid" else f"{resample}(b={_choose_block_size(m, block_size)})"
            print(f"[segment {s}:{e}] best@{s + a_rel}  stat={E_star:.4g}  "
                  f"p={pval:.3g}  via {meth}")

        # Accept / split
        if pval < significance:
            cp = s + a_rel
            accepted.append(EDivisiveSplit(
                index=cp, seg_start=s, seg_end=e,
                statistic=float(E_star), pvalue=float(pval)
            ))
            total_cps += 1
            if (max_cps is not None) and (total_cps >= max_cps):
                break
            segments.append((s, cp))
            segments.append((cp, e))
        # else: reject and do not subdivide further

    cps = np.array(sorted([sp.index for sp in accepted]), dtype=np.int64) if accepted else np.array([], dtype=np.int64)
    labels = _labels_from_cps(n, cps)
    return EDivisiveResult(n=n, alpha=float(alpha), min_size=int(min_size),
                           splits=accepted, change_points=cps, labels=labels)


def _labels_from_cps(n: int, cps: ArrayI) -> ArrayI:
    """Assign segment labels 0..K-1 given sorted change points in 1..n-1 (right-exclusive)."""
    labs = np.empty(n, dtype=int)
    prev = 0
    k = 0
    for cp in cps.tolist() + [n]:
        labs[prev:cp] = k
        k += 1
        prev = cp
    return labs
