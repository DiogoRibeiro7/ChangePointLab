from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .kcp_core import (
    KCPResult,
    build_kernel_prefix,
    gram_rbf,
    kcp_penalized,
    kcp_select_bic,
)

from ...core.datatypes import SegmentationResult
from .._base import BaseDetector

KernelFunc = Callable[..., np.ndarray | tuple[np.ndarray, float]]


@dataclass
class KernelCPD(BaseDetector):
    penalty: float
    kernel: KernelFunc = gram_rbf
    kernel_kwargs: dict[str, Any] | None = None
    min_size: int = 1
    method: str = "pelt"

    _result: KCPResult | None = None
    _kernel_gamma: float | None = None

    def fit(self, x: np.ndarray) -> KernelCPD:
        self._validate_input(x)
        kernel_out = self.kernel(x, **(self.kernel_kwargs or {}))
        if isinstance(kernel_out, tuple):
            K, gamma = kernel_out
            self._kernel_gamma = float(gamma)
        else:
            K = kernel_out
            self._kernel_gamma = None
        pref = build_kernel_prefix(K)
        self._result = kcp_penalized(
            pref,
            penalty=self.penalty,
            min_size=self.min_size,
            method=self.method,
        )
        return self

    def predict(self, x: np.ndarray | None = None) -> SegmentationResult:
        if x is not None:
            return self.fit(x).predict()
        if self._result is None:
            raise RuntimeError("Call fit before predict.")
        cps = np.array(self._result.change_points, dtype=int)
        meta = {
            "labels": self._result.labels,
            "costs": self._result.costs_per_segment,
            "edges": self._result.edges,
            "kernel_gamma": self._kernel_gamma,
        }
        return SegmentationResult(
            indices=cps,
            score=self._result.total_cost,
            labels=self._result.labels,
            method_name="kernel_cpd",
            objective_orientation="minimize",
            costs_per_segment=self._result.costs_per_segment,
            metadata=meta,
        )


__all__ = ["KernelCPD", "gram_rbf", "kcp_select_bic"]
