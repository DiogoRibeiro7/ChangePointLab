from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .edivisive_core import EDivisiveResult
from .edivisive_core import edivisive as _edivisive

from ...core.datatypes import SegmentationResult
from ...core.segmentation import normalize_linear_changepoints
from .._base import BaseDetector


@dataclass
class EDivisive(BaseDetector):
    alpha: float = 1.0
    min_size: int = 10
    R: int = 199
    seed: int | None = None

    _result: EDivisiveResult | None = None

    def fit(self, x: np.ndarray) -> EDivisive:
        self._validate_input(x)
        self._result = _edivisive(
            x, alpha=self.alpha, min_size=self.min_size, R=self.R, seed=self.seed
        )
        return self

    def predict(self, x: np.ndarray | None = None) -> SegmentationResult:
        if x is not None:
            return self.fit(x).predict()
        if self._result is None:
            raise RuntimeError("Call fit before predict.")
        cps = normalize_linear_changepoints(
            self._result.change_points,
            n=self._result.labels.size,
            min_segment_length=self.min_size,
        )
        meta = {
            "labels": self._result.labels,
            "splits": self._result.splits,
            "provenance": self._result.provenance,
        }
        return SegmentationResult(
            indices=cps,
            labels=self._result.labels,
            method_name="edivisive",
            metadata=meta,
            provenance=self._result.provenance,
        )


__all__ = ["EDivisive"]
