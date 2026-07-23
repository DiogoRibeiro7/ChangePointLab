from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from ....core.datatypes import PosteriorSampleResult
from ....core.segmentation import CircularChangePoints
from ..._base import BaseDetector

from .within_period_cpd import (
    MCMCResult,
    ModelPrior,
    RJConfig,
    Tau,
    WithinPeriodCore,
)
from .samplers import PTConfig, parallel_tempering_fit


@dataclass
class WithinPeriodCPD(BaseDetector):
    """Within-period changepoint detection via RJMCMC sampling."""

    prior: ModelPrior
    cfg: RJConfig = RJConfig()
    init: Tau | None = None
    rng: np.random.Generator | None = None

    _model: WithinPeriodCore | None = None
    _result: MCMCResult | None = None

    def fit(
        self,
        x: Sequence[int | bool],
        cfg: RJConfig | None = None,
        init: Tau | None = None,
    ) -> WithinPeriodCPD:
        """Fit the within-period model using RJMCMC."""
        x_arr = np.asarray(x, dtype=bool)
        self._validate_input(x_arr)
        self._model = WithinPeriodCore(self.prior)
        cfg = cfg or self.cfg
        init = self.init if init is None else init
        if self.rng is not None and cfg.seed is not None:
            cfg = replace(cfg, seed=None)
        self._result = self._model.fit(x_arr, cfg=cfg, init=init, rng=self.rng)
        return self

    def predict(self, x: Sequence[int | bool] | None = None) -> PosteriorSampleResult:
        if x is not None:
            return self.fit(x).predict()
        if self._result is None:
            raise RuntimeError("Call fit before predict.")
        circular = CircularChangePoints(
            period=self.prior.N,
            indices=np.array(self._result.mode_tau, dtype=int),
        )
        cps = circular.indices
        meta = {
            "cp_hist": self._result.changepoint_hist,
            "samples_tau": self._result.samples_tau,
            "log_posteriors": self._result.log_posteriors,
            "period": self.prior.N,
            "segments": circular.segments(),
            "provenance": self._result.provenance,
        }
        return PosteriorSampleResult(
            indices=cps,
            method_name="within_period",
            boundary_convention="periodic_bin_end",
            samples=tuple(tuple(sample) for sample in self._result.samples_tau),
            log_posteriors=self._result.log_posteriors,
            changepoint_hist=self._result.changepoint_hist,
            metadata=meta,
            provenance=self._result.provenance,
        )

    def __getattr__(self, name: str):
        if self._model is not None and hasattr(self._model, name):
            return getattr(self._model, name)
        raise AttributeError(name)

    @property
    def result(self) -> MCMCResult:
        if self._result is None:
            raise RuntimeError("Call fit first.")
        return self._result


__all__ = [
    "WithinPeriodCPD",
    "ModelPrior",
    "RJConfig",
    "Tau",
    "WithinPeriodCore",
    "MCMCResult",
    "PTConfig",
    "parallel_tempering_fit",
]
