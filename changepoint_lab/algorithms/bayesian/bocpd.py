from __future__ import annotations

import numpy as np

from bocpd import (
    BOCPD as _BOCPD,
)
from bocpd import (
    BOCPDConfig,
    BoostedBoundaryHazard,
    ConstantHazard,
    Hazard,
    ScheduledHazard,
)

from ...core.datatypes import ChangePointResult
from .._base import BaseDetector


class BOCPD(_BOCPD, BaseDetector):
    """Wrapper that exposes a sklearn-like interface."""

    def fit(self, x: np.ndarray) -> BOCPD:
        self._validate_input(x)
        self._result = super().run(x)
        return self

    def predict(self, x: np.ndarray | None = None) -> ChangePointResult:
        if x is not None:
            return self.fit(x).predict()
        if not hasattr(self, "_result"):
            raise RuntimeError("Call fit before predict.")
        cps = np.nonzero(self._result.cp_prob > 0.5)[0]
        meta = {
            "cp_prob": self._result.cp_prob,
            "map_run_length": self._result.map_run_length,
        }
        return ChangePointResult(indices=cps, metadata=meta)


__all__ = [
    "BOCPD",
    "BOCPDConfig",
    "Hazard",
    "ConstantHazard",
    "BoostedBoundaryHazard",
    "ScheduledHazard",
]
