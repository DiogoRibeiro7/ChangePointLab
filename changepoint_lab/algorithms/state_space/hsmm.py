from __future__ import annotations

import numpy as np

from hsmm.hsmm import HSMM as _HSMM
from hsmm.hsmm import HSMMConfig, HSMMParams, PoissonDur

from ...core.datatypes import ChangePointResult
from .._base import BaseDetector


class HSMM(_HSMM, BaseDetector):
    """HSMM wrapper exposing fit/predict methods."""

    def fit(self, loglik_tk: np.ndarray) -> HSMM:
        self._validate_input(loglik_tk)
        self.params, _ = super().fit(loglik_tk)
        self._loglik = loglik_tk
        return self

    def predict(self, loglik_tk: np.ndarray | None = None) -> ChangePointResult:
        if loglik_tk is not None:
            return self.fit(loglik_tk).predict()
        if not hasattr(self, "_loglik"):
            raise RuntimeError("Call fit before predict.")
        states, durations = super().decode_viterbi(self._loglik)
        cps = np.cumsum(durations)[:-1]
        meta = {"states": states, "durations": durations}
        return ChangePointResult(indices=cps, metadata=meta)


__all__ = ["HSMM", "HSMMConfig", "HSMMParams", "PoissonDur"]
