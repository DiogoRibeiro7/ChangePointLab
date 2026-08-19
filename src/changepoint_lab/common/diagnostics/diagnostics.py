# diagnostics.py
# MIT License

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

Tau = Tuple[int, ...]


@dataclass(frozen=True)
class PosteriorM:
    """Posterior over number of segments m."""
    m_values: NDArray[np.int64]     # unique m values, sorted ascending
    counts: NDArray[np.int64]       # occurrence counts
    probs: NDArray[np.floating]     # normalized probabilities


def posterior_num_segments(samples_tau: Sequence[Tau]) -> PosteriorM:
    """
    Compute posterior distribution of the number of segments m from RJMCMC samples.

    m = 1 if tau == (), else len(tau)+1
    """
    m_arr = np.array([1 if len(t) == 0 else (len(t) + 1) for t in samples_tau], dtype=np.int64)
    vals, counts = np.unique(m_arr, return_counts=True)
    probs = counts / counts.sum() if counts.sum() > 0 else counts.astype(float)
    return PosteriorM(m_values=vals, counts=counts, probs=probs)


def autocorr_1d(x: NDArray[np.floating], max_lag: int | None = None) -> NDArray[np.floating]:
    """
    Unbiased (approximately) autocorrelation up to max_lag (default: n//2).
    Centered & normalized by variance. Returns array of length max_lag+1 with rho[0]=1.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return np.array([1.0])
    if max_lag is None:
        max_lag = n // 2
    x = x - x.mean()
    var = x.var()
    if var == 0.0:
        return np.ones(max_lag + 1)
    rho = np.empty(max_lag + 1, dtype=float)
    rho[0] = 1.0
    for k in range(1, max_lag + 1):
        num = np.dot(x[:-k], x[k:]) / (n - k)
        rho[k] = num / var
    return rho


def ess_geyer(x: NDArray[np.floating], max_lag: int | None = None) -> float:
    """
    Effective Sample Size using Geyer's initial positive sequence (IPS).
    ESS = n / (1 + 2 * sum_{k>=1} rho_k^+), where rho_k^+ are grouped sums kept until they go non-positive.

    Notes
    -----
    - Works on a single scalar chain (1D array).
    - If variance is zero, returns +inf.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return float(n)
    r = autocorr_1d(x, max_lag=max_lag)
    if np.allclose(r, 1.0):
        return float("inf")
    # Geyer's IPS uses sums of adjacent pairs: tau_k = rho_{2k-1} + rho_{2k}
    # Keep only while tau_k > 0 and enforce monotonic decrease.
    taus: List[float] = []
    for k in range(1, (r.size // 2) + 1):
        tau_k = r[2 * k - 1] + r[2 * k] if 2 * k < r.size else r[2 * k - 1]
        if tau_k <= 0:
            break
        taus.append(tau_k)
        if len(taus) >= 2 and taus[-1] > taus[-2]:
            # enforce monotone decrease
            taus[-1] = taus[-2]
    s = np.sum(taus) if len(taus) else 0.0
    denom = 1.0 + 2.0 * s
    return float(n / denom) if denom > 0 else float(n)


def ess_for_cp_indicator(samples_tau: Sequence[Tau], N: int, max_lag: int | None = None) -> NDArray[np.floating]:
    """
    For each time-of-day index r in 0..N-1, form a binary chain 1{r in tau_t} across samples and compute ESS.
    Returns an array of length N. If a position is constant (always/never CP), ESS is +inf.
    """
    S = len(samples_tau)
    if S == 0:
        return np.zeros(N, dtype=float)
    # Build indicator matrix [S, N] lazily
    ess = np.empty(N, dtype=float)
    for r in range(N):
        chain = np.fromiter((1.0 if (r in t) else 0.0 for t in samples_tau), count=S, dtype=float)
        v = chain.var()
        ess[r] = float("inf") if v == 0.0 else ess_geyer(chain, max_lag=max_lag)
    return ess
