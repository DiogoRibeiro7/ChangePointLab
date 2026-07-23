from __future__ import annotations

import numpy as np

from .hsmm_core import HSMM as _HSMM
from .hsmm_core import HSMMConfig, HSMMParams, PoissonDur, NegBinDur

from ...core.datatypes import LatentStateResult
from ...core.segmentation import normalize_linear_changepoints
from .._base import BaseDetector


class HSMM(_HSMM, BaseDetector):
    """HSMM wrapper exposing fit/predict methods."""

    def fit(self, loglik_tk: np.ndarray) -> HSMM:
        self._validate_input(loglik_tk)
        self.params, _ = super().fit(loglik_tk)
        self._loglik = loglik_tk
        return self

    def get_params(self) -> dict[str, object]:
        """Return constructor parameters for the estimator-style wrapper."""
        return {"cfg": self.cfg, "params": self.params}

    def predict(self, loglik_tk: np.ndarray | None = None) -> LatentStateResult:
        if loglik_tk is not None:
            return self.fit(loglik_tk).predict()
        if not hasattr(self, "_loglik"):
            raise RuntimeError("Call fit before predict.")
        states, durations = super().decode_viterbi(self._loglik)
        segment_ends = np.flatnonzero(durations > 0) + 1
        cps = normalize_linear_changepoints(
            segment_ends[segment_ends < len(durations)],
            n=len(durations),
        )
        meta = {"states": states, "durations": durations}
        return LatentStateResult(
            indices=cps,
            method_name="hsmm",
            states=states,
            segment_durations=durations,
            metadata=meta,
        )


__all__ = ["HSMM", "HSMMConfig", "HSMMParams", "PoissonDur", "NegBinDur"]
