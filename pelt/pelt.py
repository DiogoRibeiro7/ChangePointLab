# pelt.py
# MIT License
# (c) 2025

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, Sequence, Tuple

import math
import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.floating]


# =========================
# Cost interface
# =========================

class SegmentCost(Protocol):
    """
    Protocol for segment cost functions C(y[a:b]) used by PELT.

    Implementations must:
      - call `precompute(y)` once before first `cost(a, b)`.
      - return the segment cost for the half-open interval [a, b) with b > a.

    Notes
    -----
    * PELT assumes the pruning condition (Eq. (4) in the paper) holds for some constant K.
      For negative log-likelihood costs, K=0 is standard. :contentReference[oaicite:1]{index=1}
    * Implementations maintain internal cumulative sums and are **not** thread-safe;
      use separate instances per concurrent execution.
    """

    def precompute(self, y: ArrayF) -> None: ...
    def cost(self, a: int, b: int) -> float: ...


# =========================
# Common cost functions
# =========================

@dataclass
class NormalMeanKnownVar(SegmentCost):
    """
    Change in mean with known variance σ^2 (Gaussian); mean re-estimated per segment.

    Cost is (up to a constant) the negative log-likelihood at the segment MLE:
        C = (SSE / σ^2) + (b-a) * log(2πσ^2)

    This is additive and satisfies Eq. (4) with K = 0. :contentReference[oaicite:2]{index=2}
    """
    sigma2: float
    _sum: Optional[ArrayF] = None
    _sum2: Optional[ArrayF] = None

    def precompute(self, y: ArrayF) -> None:
        if self.sigma2 <= 0:
            raise ValueError("sigma2 must be > 0.")
        y = np.asarray(y, dtype=float)
        self._sum = np.concatenate([[0.0], np.cumsum(y)])
        self._sum2 = np.concatenate([[0.0], np.cumsum(y * y)])

    def cost(self, a: int, b: int) -> float:
        assert self._sum is not None and self._sum2 is not None
        L = b - a
        S = float(self._sum[b] - self._sum[a])
        Q = float(self._sum2[b] - self._sum2[a])
        sse = max(Q - (S * S) / max(1, L), 0.0)
        return (sse / self.sigma2) + L * math.log(2.0 * math.pi * self.sigma2)


@dataclass
class NormalMeanVarUnknown(SegmentCost):
    """
    Gaussian with both mean and variance unknown, re-estimated per segment (standard N(μ, σ^2)).

    Cost at segment MLE (Eq. (9) in the paper):
        C = L * ( log(2π) + log(SSE / L) + 1 )
    Requires L >= 2 for a meaningful variance estimate.

    Satisfies Eq. (4) with K = 0 (negative log-likelihood additivity). :contentReference[oaicite:3]{index=3}
    """
    eps: float = 1e-12
    _sum: Optional[ArrayF] = None
    _sum2: Optional[ArrayF] = None

    def precompute(self, y: ArrayF) -> None:
        y = np.asarray(y, dtype=float)
        self._sum = np.concatenate([[0.0], np.cumsum(y)])
        self._sum2 = np.concatenate([[0.0], np.cumsum(y * y)])

    def cost(self, a: int, b: int) -> float:
        assert self._sum is not None and self._sum2 is not None
        L = b - a
        if L <= 1:
            # Penalize too-short segments heavily (so PELT avoids them when min_seg_len not enforced).
            return float("inf")
        S = float(self._sum[b] - self._sum[a])
        Q = float(self._sum2[b] - self._sum2[a])
        sse = max(Q - (S * S) / L, self.eps)
        return L * (math.log(2.0 * math.pi) + math.log(sse / L) + 1.0)


@dataclass
class BetaBinomialCost(SegmentCost):
    """
    Bernoulli(θ) with Beta(α, β) prior, marginalized (segment-wise Beta-Binomial evidence).

    Cost = -log p(y[a:b]) = -log Beta(s+α, L-s+β) + log Beta(α, β)
    where s = #ones, L = segment length.

    Again satisfies Eq. (4) with K = 0 for log-likelihood-based costs. :contentReference[oaicite:4]{index=4}
    """
    alpha: float = 1.0
    beta: float = 1.0
    _sum1: Optional[ArrayF] = None

    def precompute(self, y: ArrayF) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("alpha, beta must be > 0.")
        y = np.asarray(y, dtype=float)
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("Input y must be binary {0,1} for BetaBinomialCost.")
        self._sum1 = np.concatenate([[0.0], np.cumsum(y)])

    def cost(self, a: int, b: int) -> float:
        assert self._sum1 is not None
        L = b - a
        s = float(self._sum1[b] - self._sum1[a])
        # -log Beta(s+α, L-s+β) + log Beta(α, β)
        return -_log_beta(s + self.alpha, (L - s) + self.beta) + _log_beta(self.alpha, self.beta)


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


# =========================
# Penalty helpers
# =========================

def bic_penalty(params_per_segment: int, n: int) -> float:
    """Schwarz (SIC/BIC) penalty per changepoint: β = p * log n."""
    return params_per_segment * math.log(max(2, n))


def aic_penalty(params_per_segment: int) -> float:
    """AIC penalty per changepoint: β = 2p."""
    return 2.0 * params_per_segment


# =========================
# Result container
# =========================

@dataclass
class PELTResult:
    """
    Outputs of PELT.

    Attributes
    ----------
    change_points : list[int]
        Sorted changepoint locations in {1, ..., n-1}. Empty if no change.
    total_cost : float
        Objective value: sum_i C(segment_i) + β * m  (m = #changepoints).
    F : ArrayF
        DP values F[t] for t=0..n (F[0] = -β).
    prev : NDArray[np.int64]
        Backpointers: prev[t] = last change location before t (argmin at t).
    """
    change_points: List[int]
    total_cost: float
    F: ArrayF
    prev: NDArray[np.int64]


# =========================
# PELT core (Alg. 2)
# =========================

def pelt(
    y: Sequence[float],
    cost_fn: SegmentCost,
    *,
    penalty: float,
    min_seg_len: int = 1,
    K: float = 0.0,
) -> PELTResult:
    """
    Pruned Exact Linear Time changepoint detection (PELT). Exact minimization of
        sum_i C(segment_i) + β m,
    using pruning condition (Alg. 2, Thm. 3.1) with constant K (often 0). :contentReference[oaicite:5]{index=5}

    Parameters
    ----------
    y : sequence of floats
        Data sequence length n.
    cost_fn : SegmentCost
        Segment cost object implementing `precompute` and `cost(a, b)`.
    penalty : float
        Linear penalty β per changepoint.
    min_seg_len : int, default=1
        Minimum segment length (enforced exactly).
    K : float, default=0.0
        Constant for the pruning inequality (Eq. (4)). For (penalized) log-likelihood
        costs, K=0 is standard.

    Returns
    -------
    PELTResult
    """
    y_arr = np.asarray(y, dtype=float)
    n = int(y_arr.size)
    if n == 0:
        return PELTResult(change_points=[], total_cost=0.0,
                          F=np.array([0.0]), prev=np.array([-1], dtype=np.int64))
    if min_seg_len < 1 or min_seg_len > n:
        raise ValueError("min_seg_len must be in [1, n].")
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be non-negative and finite.")

    # Precompute cost stats
    cost_fn.precompute(y_arr)

    # DP arrays
    F = np.full(n + 1, float("inf"))
    prev = np.full(n + 1, -1, dtype=np.int64)
    F[0] = -penalty  # as in OP/PELT to make total penalty = m*β. :contentReference[oaicite:6]{index=6}

    # Candidate set R_t (possible last-change positions)
    R: List[int] = [0]

    # Main loop
    for t in range(min_seg_len, n + 1):  # t is end index (exclusive), segment is [τ, t)
        # Eligible candidates (respect min_seg_len)
        Rt = [τ for τ in R if (t - τ) >= min_seg_len]

        # Evaluate DP objective for each candidate τ
        best_val = float("inf")
        best_tau = -1
        for τ in Rt:
            val = F[τ] + cost_fn.cost(τ, t) + penalty
            if val < best_val:
                best_val = val
                best_tau = τ

        F[t] = best_val
        prev[t] = best_tau

        # Update candidate set for t+1:
        # add the latest index that *can* be a last-change at t+1: τ_new = t+1 - min_seg_len
        τ_new = t + 1 - min_seg_len
        if 0 <= τ_new <= n:
            Rt_plus = Rt + [τ_new]
        else:
            Rt_plus = Rt

        # Prune (Thm. 3.1): keep τ with F[τ] + C(τ, t) + K <= F[t]
        R = []
        for τ in Rt_plus:
            lhs = F[τ] + cost_fn.cost(τ, t) + K
            if lhs <= F[t] + 1e-12:  # tiny slack for floating error
                R.append(τ)

    # Backtrack from t=n to recover changepoints
    cps: List[int] = []
    t = n
    while t > 0:
        τ = int(prev[t])
        if τ < 0:
            # No valid backpointer: this can occur if min_seg_len > n; guard above prevents it.
            break
        if τ > 0:
            cps.append(τ)
        t = τ
    cps.reverse()

    return PELTResult(change_points=cps, total_cost=F[n], F=F, prev=prev)


# =========================
# Concave-penalty wrapper (Sec. 3.2)
# =========================

def pelt_concave_penalty(
    y: Sequence[float],
    cost_fn: SegmentCost,
    *,
    f: Callable[[int], float],
    fprime: Callable[[int], float],
    min_seg_len: int = 1,
    K: float = 0.0,
    max_iter: int = 20,
) -> PELTResult:
    """
    Handle penalties of the form β f(m) with f concave & differentiable (Sec. 3.2).
    Iterates PELT with β = f'(m_hat) until the number of changes stabilizes. :contentReference[oaicite:7]{index=7}

    Parameters
    ----------
    y : sequence of floats
    cost_fn : SegmentCost
    f, fprime : concave penalty and its derivative
    min_seg_len : int
    K : float
    max_iter : int

    Returns
    -------
    PELTResult
        Result from final iteration (β = f'(m_hat)).
    """
    # Initialize at derivative near m=1
    m_old = -1
    m_hat = 1
    res: Optional[PELTResult] = None

    for _ in range(max_iter):
        beta = float(fprime(max(1, m_hat)))
        res = pelt(y, cost_fn, penalty=beta, min_seg_len=min_seg_len, K=K)
        m_new = len(res.change_points)
        if m_new == m_old:
            break
        m_old, m_hat = m_hat, m_new

    assert res is not None
    return res
