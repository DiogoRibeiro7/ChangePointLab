from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .kcp_rff import RFFConfig, RFFKCPResult, build_feature_prefix, rbf_rff_map, rff_kcp_penalized
from .kcp_core import (
    KCPResult,
    build_kernel_prefix,
    gram_rbf,
    kcp_penalized,
    kcp_select_bic,
)

from ...core.datatypes import SegmentationResult
from ...core.segmentation import normalize_linear_changepoints
from .._base import BaseDetector


@dataclass(frozen=True)
class KernelMatrix:
    """Typed output for callables that return a Gram matrix plus metadata."""

    gram: np.ndarray
    metadata: dict[str, Any] | None = None


KernelFunc = Callable[..., np.ndarray | tuple[np.ndarray, float] | KernelMatrix]


@dataclass
class KernelCPD(BaseDetector):
    penalty: float
    kernel: KernelFunc = gram_rbf
    kernel_kwargs: dict[str, Any] | None = None
    min_size: int = 1
    method: str = "pelt"
    grid_jump: int = 1
    max_seg_len: int | None = None
    bandwidth: float | None = None
    approximation: str = "exact"
    rff_config: RFFConfig | None = None
    kernel_psd_tol: float = 1e-8
    max_gram_bytes: int | None = None

    _result: KCPResult | RFFKCPResult | None = None
    _kernel_gamma: float | None = None
    _kernel_metadata: dict[str, Any] | None = None
    _approximation: str = "exact"

    def _validate_config(self) -> None:
        if not np.isfinite(self.penalty) or self.penalty < 0:
            raise ValueError("penalty must be a non-negative finite number.")
        if self.min_size < 1:
            raise ValueError("min_size must be >= 1.")
        if self.method not in {"pelt", "op"}:
            raise ValueError("method must be 'pelt' or 'op'.")
        if self.grid_jump < 1:
            raise ValueError("grid_jump must be >= 1.")
        if self.max_seg_len is not None and self.max_seg_len < self.min_size:
            raise ValueError("max_seg_len must be >= min_size.")
        if self.bandwidth is not None and (
            not np.isfinite(self.bandwidth) or self.bandwidth <= 0
        ):
            raise ValueError("bandwidth must be a positive finite number.")
        if self.approximation not in {"exact", "rff"}:
            raise ValueError("approximation must be 'exact' or 'rff'.")
        if self.max_gram_bytes is not None and self.max_gram_bytes < 1:
            raise ValueError("max_gram_bytes must be positive when provided.")
        if self.kernel_psd_tol < 0 or not np.isfinite(self.kernel_psd_tol):
            raise ValueError("kernel_psd_tol must be a non-negative finite number.")

    def _kernel_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self.kernel_kwargs or {})
        if self.bandwidth is not None:
            kwargs.setdefault("sigma", self.bandwidth)
        return kwargs

    @staticmethod
    def _normalize_kernel_output(output: np.ndarray | tuple[np.ndarray, float] | KernelMatrix) -> KernelMatrix:
        if isinstance(output, KernelMatrix):
            return output
        if isinstance(output, tuple):
            if len(output) != 2:
                raise ValueError("kernel tuple outputs must be (gram, gamma).")
            gram, gamma = output
            return KernelMatrix(gram=gram, metadata={"kernel_gamma": float(gamma)})
        return KernelMatrix(gram=output, metadata={})

    def fit(self, x: np.ndarray) -> KernelCPD:
        self._validate_input(x)
        self._validate_config()
        self._approximation = "rff" if self.rff_config is not None else self.approximation
        if self._approximation == "rff":
            cfg = self.rff_config or RFFConfig()
            rff = rbf_rff_map(x, cfg=cfg, sigma=self.bandwidth)
            pref = build_feature_prefix(rff.Z)
            self._result = rff_kcp_penalized(
                pref,
                gamma_pen=self.penalty,
                min_size=self.min_size,
                method=self.method,
                grid_jump=self.grid_jump,
                max_seg_len=self.max_seg_len,
                rff_gamma=rff.gamma,
            )
            self._kernel_gamma = rff.gamma
            self._kernel_metadata = {
                "kernel": "rbf",
                "kernel_gamma": rff.gamma,
                "bandwidth": self.bandwidth,
                "approximation": "rff",
                "rff_n_features": cfg.n_features,
                "rff_seed": cfg.seed,
                "rff_subsample_for_bandwidth": cfg.subsample_for_bandwidth,
            }
        else:
            kernel_out = self.kernel(x, **self._kernel_kwargs())
            kernel_matrix = self._normalize_kernel_output(kernel_out)
            metadata = dict(kernel_matrix.metadata or {})
            self._kernel_gamma = metadata.get("kernel_gamma")
            if self._kernel_gamma is not None:
                self._kernel_gamma = float(self._kernel_gamma)
            pref = build_kernel_prefix(
                kernel_matrix.gram,
                psd_tol=self.kernel_psd_tol,
                max_bytes=self.max_gram_bytes,
            )
            self._result = kcp_penalized(
                pref,
                penalty=self.penalty,
                min_size=self.min_size,
                method=self.method,
                grid_jump=self.grid_jump,
                max_seg_len=self.max_seg_len,
            )
            self._kernel_metadata = {
                **metadata,
                "bandwidth": self.bandwidth,
                "approximation": "exact",
            }
        return self

    def predict(self, x: np.ndarray | None = None) -> SegmentationResult:
        if x is not None:
            return self.fit(x).predict()
        if self._result is None:
            raise RuntimeError("Call fit before predict.")
        cps = normalize_linear_changepoints(
            self._result.change_points,
            n=self._result.n,
            min_segment_length=self.min_size,
        )
        meta = {
            "labels": self._result.labels,
            "costs": self._result.costs_per_segment,
            "edges": self._result.edges,
            "kernel_gamma": self._kernel_gamma,
            "kernel_metadata": self._kernel_metadata or {},
            "approximation": self._approximation,
            "method": self.method,
            "min_size": self.min_size,
            "grid_jump": self.grid_jump,
            "max_seg_len": self.max_seg_len,
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


__all__ = ["KernelCPD", "KernelMatrix", "RFFConfig", "gram_rbf", "kcp_select_bic"]
