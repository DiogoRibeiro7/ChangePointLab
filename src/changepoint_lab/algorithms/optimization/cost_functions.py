from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Optional

import math
import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.floating[Any]]


class SegmentCost(Protocol):
    """Protocol for segment cost functions used by PELT."""

    def precompute(self, y: ArrayF) -> None: ...
    def cost(self, a: int, b: int) -> float: ...


@dataclass
class NormalMeanKnownVar(SegmentCost):
    """Gaussian mean shifts with known variance.

    The returned segment cost is the profile deviance, i.e. twice the negative
    Gaussian log-likelihood after optimizing the segment mean.
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
    """Gaussian mean/variance shifts with unknown parameters.

    The returned segment cost is the profile deviance, i.e. twice the negative
    Gaussian log-likelihood after optimizing the segment mean and variance.
    Segments of length one have undefined variance and return infinity.
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
            return float("inf")
        S = float(self._sum[b] - self._sum[a])
        Q = float(self._sum2[b] - self._sum2[a])
        sse = max(Q - (S * S) / L, self.eps)
        return L * (math.log(2.0 * math.pi) + math.log(sse / L) + 1.0)


@dataclass
class BetaBinomialCost(SegmentCost):
    """Bernoulli segments with a marginalized Beta prior.

    The returned segment cost is the negative log Beta-binomial marginal
    likelihood. It is not on the Gaussian deviance scale used by the AIC/BIC
    penalty helpers.
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
        return -_log_beta(s + self.alpha, (L - s) + self.beta) + _log_beta(self.alpha, self.beta)


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def bic_penalty(params_per_segment: int, n: int) -> float:
    """Schwarz (BIC) penalty per changepoint in deviance units."""
    return params_per_segment * math.log(max(2, n))


def aic_penalty(params_per_segment: int) -> float:
    """AIC penalty per changepoint in deviance units."""
    return 2.0 * params_per_segment


__all__ = [
    "SegmentCost",
    "NormalMeanKnownVar",
    "NormalMeanVarUnknown",
    "BetaBinomialCost",
    "bic_penalty",
    "aic_penalty",
]
