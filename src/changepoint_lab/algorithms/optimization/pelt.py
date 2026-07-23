from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...core.datatypes import ChangePointResult, SegmentationResult
from .._base import BaseDetector
from .cost_functions import (
    BetaBinomialCost,
    NormalMeanKnownVar,
    NormalMeanVarUnknown,
    SegmentCost,
    aic_penalty,
    bic_penalty,
)

# Scientific traceability:
# - Killick, Fearnhead, and Eckley (2012), doi:10.1080/01621459.2012.737745.
# - Registry entry: docs/science/method_registry.yml, method id "pelt".

ArrayF = NDArray[np.floating]
ArrayI = NDArray[np.int_]


@dataclass
class PELTResult:
    change_points: list[int]
    total_cost: float
    F: ArrayF
    prev: NDArray[np.int64]
    labels: ArrayI
    costs_per_segment: ArrayF


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
    """Pruned Exact Linear Time changepoint detection."""
    y_arr = np.asarray(y, dtype=float)
    n = int(y_arr.size)
    if n == 0:
        return PELTResult(
            change_points=[],
            total_cost=0.0,
            F=np.array([0.0]),
            prev=np.array([-1], dtype=np.int64),
            labels=np.array([], dtype=int),
            costs_per_segment=np.array([], dtype=float),
        )
    if min_seg_len < 1 or min_seg_len > n:
        raise ValueError("min_seg_len must be in [1, n].")
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be non-negative and finite.")

    cost_fn.precompute(y_arr)

    F = np.full(n + 1, float("inf"))
    prev = np.full(n + 1, -1, dtype=np.int64)
    F[0] = -penalty

    R = deque([0])

    for t in range(min_seg_len, n + 1):
        cost_cache = {}
        eligible: list[int] = []
        best_val = float("inf")
        best_tau = -1
        for tau in R:
            if t - tau < min_seg_len:
                continue
            c = cost_fn.cost(tau, t)
            cost_cache[tau] = c
            eligible.append(tau)
            val = F[tau] + c + penalty
            if val < best_val:
                best_val = val
                best_tau = tau
        F[t] = best_val
        prev[t] = best_tau

        tau_new = t + 1 - min_seg_len
        prune_candidates = eligible.copy()
        if 0 <= tau_new <= n:
            c_new = cost_fn.cost(tau_new, t)
            cost_cache[tau_new] = c_new
            prune_candidates.append(tau_new)

        new_R = deque()
        for tau in prune_candidates:
            if F[tau] + cost_cache[tau] + K <= F[t] + 1e-12:
                new_R.append(tau)
        R = new_R

    cps: list[int] = []
    t = n
    while t > 0:
        tau = int(prev[t])
        if tau < 0:
            break
        if tau > 0:
            cps.append(tau)
        t = tau
    cps.reverse()
    edges = [0] + cps + [n]
    labels = np.empty(n, dtype=int)
    costs = []
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        labels[a:b] = k
        costs.append(cost_fn.cost(a, b))
    return PELTResult(
        change_points=cps,
        total_cost=F[n],
        F=F,
        prev=prev,
        labels=labels,
        costs_per_segment=np.asarray(costs, dtype=float),
    )


# =========================
# Concave-penalty wrapper
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
    m_old = -1
    m_hat = 1
    res: PELTResult | None = None
    for _ in range(max_iter):
        beta = float(fprime(max(1, m_hat)))
        res = pelt(y, cost_fn, penalty=beta, min_seg_len=min_seg_len, K=K)
        m_new = len(res.change_points)
        if m_new == m_old:
            break
        m_old, m_hat = m_hat, m_new
    assert res is not None
    return res


@dataclass
class PELT(BaseDetector):
    cost_fn: SegmentCost
    penalty: float
    min_seg_len: int = 1

    _result: PELTResult | None = None

    def fit(self, x: np.ndarray) -> PELT:
        self._validate_input(x)
        self._result = pelt(x, self.cost_fn, penalty=self.penalty, min_seg_len=self.min_seg_len)
        return self

    def predict(self, x: np.ndarray | None = None) -> SegmentationResult:
        if x is not None:
            return self.fit(x).predict()
        if self._result is None:
            raise RuntimeError("Call fit before predict.")
        cps = np.array(self._result.change_points, dtype=int)
        meta = {
            "labels": self._result.labels,
            "costs_per_segment": self._result.costs_per_segment,
        }
        return SegmentationResult(
            indices=cps,
            score=self._result.total_cost,
            labels=self._result.labels,
            method_name="pelt",
            objective_orientation="minimize",
            costs_per_segment=self._result.costs_per_segment,
            metadata=meta,
        )


def pelt_detect(
    x: Sequence[float],
    cost_fn: SegmentCost,
    *,
    penalty: float,
    min_seg_len: int = 1,
) -> ChangePointResult:
    model = PELT(cost_fn=cost_fn, penalty=penalty, min_seg_len=min_seg_len)
    return model.fit_predict(np.asarray(x, dtype=float))


__all__ = [
    "PELT",
    "PELTResult",
    "pelt",
    "pelt_concave_penalty",
    "pelt_detect",
    "SegmentCost",
    "NormalMeanKnownVar",
    "NormalMeanVarUnknown",
    "BetaBinomialCost",
    "bic_penalty",
    "aic_penalty",
]
