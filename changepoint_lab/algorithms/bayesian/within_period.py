from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .._base import BaseDetector
from ...core.datatypes import ChangePointResult
from within_period.within_period_cpd import (
    WithinPeriodCPD,
    ModelPrior,
    RJConfig,
    Tau,
    MCMCResult,
)


@dataclass
class WithinPeriodBOCPD(BaseDetector):
    prior: ModelPrior
    cfg: RJConfig = RJConfig()
    init: Optional[Tau] = None

    _result: Optional[MCMCResult] = None

    def fit(self, x: np.ndarray) -> "WithinPeriodBOCPD":
        self._validate_input(x)
        model = WithinPeriodCPD(self.prior)
        self._result = model.fit(x, cfg=self.cfg, init=self.init)
        return self

    def predict(self, x: Optional[np.ndarray] = None) -> ChangePointResult:
        if x is not None:
            return self.fit(x).predict()
        if self._result is None:
            raise RuntimeError("Call fit before predict.")
        cps = np.array(self._result.mode_tau, dtype=int)
        meta = {"cp_hist": self._result.changepoint_hist}
        return ChangePointResult(indices=cps, metadata=meta)


__all__ = ["WithinPeriodBOCPD", "ModelPrior", "RJConfig"]
