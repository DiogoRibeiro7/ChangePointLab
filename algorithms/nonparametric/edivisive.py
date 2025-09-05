from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from algorithms._base import BaseDetector
from core.datatypes import ChangePointResult
from edivisive.edivisive import edivisive as _edivisive, EDivisiveResult


@dataclass
class EDivisive(BaseDetector):
    alpha: float = 1.0
    min_size: int = 10
    R: int = 199
    seed: Optional[int] = None

    _result: Optional[EDivisiveResult] = None

    def fit(self, x: np.ndarray) -> "EDivisive":
        self._validate_input(x)
        self._result = _edivisive(
            x, alpha=self.alpha, min_size=self.min_size, R=self.R, seed=self.seed
        )
        return self

    def predict(self, x: Optional[np.ndarray] = None) -> ChangePointResult:
        if x is not None:
            return self.fit(x).predict()
        if self._result is None:
            raise RuntimeError("Call fit before predict.")
        cps = np.array(self._result.change_points, dtype=int)
        meta = {"labels": self._result.labels, "splits": self._result.splits}
        return ChangePointResult(indices=cps, metadata=meta)


__all__ = ["EDivisive"]
