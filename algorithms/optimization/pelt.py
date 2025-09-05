from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from algorithms._base import BaseDetector, ChangePointResult
from pelt.pelt import (
    pelt as _pelt,
    SegmentCost,
    NormalMeanKnownVar,
    NormalMeanVarUnknown,
    BetaBinomialCost,
    PELTResult,
)


@dataclass
class PELT(BaseDetector):
    cost_fn: SegmentCost
    penalty: float
    min_seg_len: int = 1

    _result: Optional[PELTResult] = None

    def fit(self, x: np.ndarray) -> "PELT":
        self._validate_input(x)
        self.cost_fn.precompute(x)
        self._result = _pelt(x, self.cost_fn, penalty=self.penalty, min_seg_len=self.min_seg_len)
        return self

    def predict(self, x: Optional[np.ndarray] = None) -> ChangePointResult:
        if x is not None:
            return self.fit(x).predict()
        if self._result is None:
            raise RuntimeError("Call fit before predict.")
        cps = np.array(self._result.change_points, dtype=int)
        meta = {
            "labels": self._result.labels,
            "costs_per_segment": self._result.costs_per_segment,
        }
        return ChangePointResult(indices=cps, score=self._result.total_cost, metadata=meta)


__all__ = [
    "PELT",
    "SegmentCost",
    "NormalMeanKnownVar",
    "NormalMeanVarUnknown",
    "BetaBinomialCost",
]
