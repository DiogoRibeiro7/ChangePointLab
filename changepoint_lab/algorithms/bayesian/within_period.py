from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from within_period.within_period_cpd import (
    MCMCResult,
    ModelPrior,
    RJConfig,
    Tau,
    WithinPeriodCPD,
)

from ...core.datatypes import ChangePointResult
from .._base import BaseDetector


@dataclass
class WithinPeriodBOCPD(BaseDetector):
    prior: ModelPrior
    cfg: RJConfig = RJConfig()
    init: Tau | None = None

    _result: MCMCResult | None = None

    def fit(self, x: np.ndarray) -> WithinPeriodBOCPD:
        self._validate_input(x)
        model = WithinPeriodCPD(self.prior)
        self._result = model.fit(x, cfg=self.cfg, init=self.init)
        return self

    def predict(self, x: np.ndarray | None = None) -> ChangePointResult:
        if x is not None:
            return self.fit(x).predict()
        if self._result is None:
            raise RuntimeError("Call fit before predict.")
        cps = np.array(self._result.mode_tau, dtype=int)
        meta = {"cp_hist": self._result.changepoint_hist}
        return ChangePointResult(indices=cps, metadata=meta)


__all__ = ["WithinPeriodBOCPD", "ModelPrior", "RJConfig"]
