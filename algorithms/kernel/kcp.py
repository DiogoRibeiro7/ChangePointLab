from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Any

import numpy as np

from algorithms._base import BaseDetector, ChangePointResult
from kcp.kcp import (
    kcp_penalized,
    kcp_select_bic,
    gram_rbf,
    KCPResult,
)

KernelFunc = Callable[[np.ndarray], np.ndarray]


@dataclass
class KernelCPD(BaseDetector):
    penalty: float
    kernel: Callable[[np.ndarray], np.ndarray] = gram_rbf
    kernel_kwargs: dict[str, Any] | None = None

    _result: Optional[KCPResult] = None

    def fit(self, x: np.ndarray) -> "KernelCPD":
        self._validate_input(x)
        K = self.kernel(x, **(self.kernel_kwargs or {}))
        self._result = kcp_penalized(K, penalty=self.penalty)
        return self

    def predict(self, x: Optional[np.ndarray] = None) -> ChangePointResult:
        if x is not None:
            return self.fit(x).predict()
        if self._result is None:
            raise RuntimeError("Call fit before predict.")
        cps = np.array(self._result.change_points, dtype=int)
        meta = {"labels": self._result.labels, "costs": self._result.costs_per_segment}
        return ChangePointResult(indices=cps, score=self._result.total_cost, metadata=meta)


__all__ = ["KernelCPD", "gram_rbf", "kcp_select_bic"]
