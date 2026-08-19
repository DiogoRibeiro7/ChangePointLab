from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np

from .core import (
    BOCPD as _BOCPD,
    BOCPDAlertConfig,
    BOCPDConfig,
    BOCPDResult,
    BoostedBoundaryHazard,
    ConstantHazard,
    Hazard,
    ScheduledHazard,
    extract_changepoint_alerts,
)
from .likelihoods import BetaBernoulli, ConjugateLikelihood, PoissonGamma
from ....core.datatypes import OnlineProbabilityResult
from ..._base import BaseDetector


class BOCPD(_BOCPD, BaseDetector):
    """Wrapper that exposes a sklearn-like interface."""

    def fit(self, x: np.ndarray) -> BOCPD:
        self._validate_input(x)
        self._result = super().run(cast("Sequence[object]", x))
        return self

    def get_params(self) -> dict[str, object]:
        """Return constructor parameters for the estimator-style wrapper."""
        return {"hazard": self.hazard, "cfg": self.cfg, "likelihood": self.lik}

    def predict(self, x: np.ndarray | None = None) -> OnlineProbabilityResult:
        if x is not None:
            return self.fit(x).predict()
        if not hasattr(self, "_result"):
            raise RuntimeError("Call fit before predict.")
        cps = extract_changepoint_alerts(self._result, self.cfg.alert_config)
        meta = {
            "cp_prob": self._result.cp_prob,
            "map_run_length": self._result.map_run_length,
            "log_evidence": self._result.log_evidence,
            "approximation_error": self._result.approximation_error,
            "alert_config": self.cfg.alert_config,
            "diagnostics": self._result.diagnostics,
        }
        return OnlineProbabilityResult(
            indices=cps,
            method_name="bocpd",
            boundary_convention="time_index",
            cp_prob=self._result.cp_prob,
            map_run_length=self._result.map_run_length,
            pred_mean=self._result.pred_mean,
            metadata=meta,
        )


def plot_run_length_heatmap(*args, **kwargs):
    """Plot a BOCPD run-length posterior if the plotting extra is installed."""
    from .plotting import plot_run_length_heatmap as _plot_run_length_heatmap

    return _plot_run_length_heatmap(*args, **kwargs)


def plot_cp_probability(*args, **kwargs):
    """Plot BOCPD changepoint probabilities if the plotting extra is installed."""
    from .plotting import plot_cp_probability as _plot_cp_probability

    return _plot_cp_probability(*args, **kwargs)


__all__ = [
    "BOCPD",
    "BOCPDAlertConfig",
    "BOCPDConfig",
    "BOCPDResult",
    "ConjugateLikelihood",
    "BetaBernoulli",
    "PoissonGamma",
    "Hazard",
    "ConstantHazard",
    "BoostedBoundaryHazard",
    "ScheduledHazard",
    "extract_changepoint_alerts",
    "plot_run_length_heatmap",
    "plot_cp_probability",
]
