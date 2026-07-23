# bocpd.py
# MIT License
# (c) 2025

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Set
import warnings

import numpy as np
from numpy.typing import NDArray

from .likelihoods import BetaBernoulli, ConjugateLikelihood
from .validation import int_ge, positive_float

# Scientific traceability:
# - Adams and MacKay (2007), Bayesian Online Changepoint Detection,
#   https://arxiv.org/abs/0710.3742.
# - Registry entry: docs/science/method_registry.yml, method id "bocpd_beta_bernoulli".

# -------------------------- Type aliases ---------------------------

ArrayF = NDArray[np.floating]
ArrayB = NDArray[np.bool_]


# -------------------------- Hazards API ----------------------------

class Hazard:
    """Protocol-like base: any object with .prob(r: int, t: int) -> float is a hazard."""

    def prob(self, r: int, t: int) -> float:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantHazard:
    """Memoryless hazard with mean run length λ: ``h = 1 / λ``."""

    mean_run_length: float = 96.0
    eps: float = 1e-12

    def __post_init__(self) -> None:
        positive_float(self.mean_run_length, "mean_run_length")
        positive_float(self.eps, "eps")

    def prob(self, r: int, t: int) -> float:
        if self.mean_run_length <= 0:
            raise ValueError("mean_run_length must be > 0.")
        h = 1.0 / float(self.mean_run_length)
        # Clamp strictly inside (0,1) to avoid degeneracy
        return float(np.clip(h, self.eps, 1.0 - self.eps))


@dataclass(frozen=True)
class ScheduledHazard:
    """
    Periodic hazard schedule: h_t = schedule[t % period].

    Parameters
    ----------
    schedule : 1-D array-like of length == period, entries in (0,1)
    period   : int > 0
    """
    schedule: ArrayF
    period: int

    def __post_init__(self) -> None:
        int_ge(self.period, "period", 1)
        # Ensure numpy array (dataclass is frozen; use object.__setattr__)
        sched = np.asarray(self.schedule, dtype=float)
        object.__setattr__(self, "schedule", sched)
        if sched.ndim != 1 or sched.size != self.period:
            raise ValueError("schedule must be 1-D with length == period")
        if not np.all((sched > 0.0) & (sched < 1.0)):
            raise ValueError("schedule entries must be in (0,1)")

    def prob(self, r: int, t: int) -> float:
        return float(self.schedule[int(t % self.period)])


@dataclass(frozen=True)
class BoostedBoundaryHazard:
    """Boost a base hazard on specified period boundaries."""

    base: Hazard
    period: int
    boundaries: Set[int]
    boost_factor: float = 10.0
    eps: float = 1e-12

    def __init__(
        self,
        base: Hazard,
        *,
        period: int,
        boundaries: Set[int] | None = None,
        boost_factor: float = 10.0,
        eps: float = 1e-12,
        boundary_indices: Set[int] | None = None,
    ) -> None:
        if boundaries is None:
            boundaries = boundary_indices if boundary_indices is not None else set()
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "period", period)
        object.__setattr__(self, "boundaries", set(boundaries))
        object.__setattr__(self, "boost_factor", boost_factor)
        object.__setattr__(self, "eps", eps)
        self.__post_init__()

    def __post_init__(self) -> None:
        int_ge(self.period, "period", 1)
        positive_float(self.boost_factor, "boost_factor")
        bad = [i for i in self.boundaries if not (0 <= int(i) < self.period)]
        if bad:
            raise ValueError(
                f"boundaries out of range for period={self.period}: {bad}"
            )

    def prob(self, r: int, t: int) -> float:
        h = float(self.base.prob(r, t))
        if (t % self.period) in self.boundaries:
            h *= self.boost_factor
        return float(np.clip(h, self.eps, 1.0 - self.eps))


# ----------------------- Config & result types ----------------------

@dataclass(frozen=True)
class BOCPDAlertConfig:
    """Post-processing policy for extracting BOCPD changepoint alerts."""

    probability_threshold: float | None = None
    require_local_peak: bool = True
    use_run_length_reset: bool = False
    min_spacing: int = 1

    def __post_init__(self) -> None:
        if self.probability_threshold is not None and not (
            0.0 <= self.probability_threshold <= 1.0
        ):
            raise ValueError("probability_threshold must be in [0, 1].")
        int_ge(self.min_spacing, "min_spacing", 1)


@dataclass(frozen=True)
class BOCPDConfig:
    """
    Configuration for the BOCPD solver.

    Attributes
    ----------
    alpha0, beta0 : float
        Beta prior hyperparameters for the Bernoulli likelihood.
    max_run_length : int
        Maximum run length tracked (posterior vector size is R = max_run_length + 1).
    store_run_length_posterior : bool
        If True, store per-step run-length posterior (for heatmaps).
    prune_epsilon : float
        Threshold for tail-pruning tiny probabilities after each update.
        If prune_relative is True, effective threshold is eps * max(R_next).
    prune_relative : bool
        If True, use relative pruning; else use absolute pruning.
    stabilizer : float
        Small positive floor used to detect underflow during normalization.
    top_k : Optional[int]
        If set, keep only the K largest run-length states (plus r=0) each step.
    cp_scale : float
        Deprecated compatibility multiplier for changepoint transition mass.
        Values other than 1.0 intentionally produce a scaled alert score, not a
        canonical posterior probability.
    alert_config : BOCPDAlertConfig
        Explicit post-processing policy for wrapper-level changepoint alerts.
    """
    alpha0: float = 1.0
    beta0: float = 1.0
    max_run_length: int = 512
    store_run_length_posterior: bool = True
    prune_epsilon: float = 0.0
    prune_relative: bool = True
    stabilizer: float = 1e-300
    top_k: Optional[int] = None
    cp_scale: float = 1.0
    alert_config: BOCPDAlertConfig = field(default_factory=BOCPDAlertConfig)

    def __post_init__(self) -> None:
        positive_float(self.alpha0, "alpha0")
        positive_float(self.beta0, "beta0")
        int_ge(self.max_run_length, "max_run_length", 1)
        if self.prune_epsilon < 0.0:
            raise ValueError("prune_epsilon must be >= 0")
        positive_float(self.stabilizer, "stabilizer")
        if self.top_k is not None:
            int_ge(self.top_k, "top_k", 1)
        positive_float(self.cp_scale, "cp_scale")
        if self.cp_scale != 1.0:
            warnings.warn(
                "BOCPDConfig.cp_scale is deprecated because it changes the "
                "run-length posterior normalization. Use a hazard model or "
                "BOCPDAlertConfig for alerting policy instead.",
                DeprecationWarning,
                stacklevel=2,
            )


@dataclass
class BOCPDResult:
    """
    Results from processing a sequence.

    Attributes
    ----------
    cp_prob : ArrayF, shape (T,)
        P(r_t = 0 | x_{1:t}) at each time.
    map_run_length : NDArray[np.int_], shape (T,)
        Argmax run-length at each time t.
    pred_mean : ArrayF, shape (T,)
        One-step-ahead predictive mean P(x_t=1 | x_{1:t-1}) before seeing x_t.
    run_length_posterior : Optional[ArrayF], shape (T, R)
        Posterior over run-length if requested.
    log_evidence : Optional[ArrayF], shape (T,)
        Per-step log normalizer before pruning/top-k approximation.
    approximation_error : Optional[ArrayF], shape (T,)
        Per-step removed probability mass from truncation, pruning, and top-k.
    diagnostics : Mapping[str, Any]
        Numerical and compatibility diagnostics.
    """
    cp_prob: ArrayF
    map_run_length: NDArray[np.int_]
    pred_mean: ArrayF
    run_length_posterior: Optional[ArrayF]
    log_evidence: Optional[ArrayF] = None
    approximation_error: Optional[ArrayF] = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        T = len(self.cp_prob)
        if len(self.map_run_length) != T or len(self.pred_mean) != T:
            raise ValueError("cp_prob, map_run_length and pred_mean must have equal length")
        if self.run_length_posterior is not None and self.run_length_posterior.shape[0] != T:
            raise ValueError("run_length_posterior must have first dimension == len(cp_prob)")
        if self.log_evidence is not None and len(self.log_evidence) != T:
            raise ValueError("log_evidence must have length == len(cp_prob)")
        if self.approximation_error is not None and len(self.approximation_error) != T:
            raise ValueError("approximation_error must have length == len(cp_prob)")


# ------------------------------ Model ---------------------------------

class BOCPD:
    """
    Bayesian Online Changepoint Detection (Adams & MacKay, 2007) with a
    pluggable scalar conjugate likelihood.

    Hazard H(r, t) controls the prior CP probability; use ConstantHazard(λ) for the classic model.
    """

    def __init__(
        self,
        hazard: Hazard,
        cfg: BOCPDConfig = BOCPDConfig(),
        *,
        likelihood: ConjugateLikelihood | None = None,
    ) -> None:
        self.hazard = hazard
        self.cfg = cfg
        self._likelihood_template = (
            likelihood.clone()
            if likelihood is not None
            else BetaBernoulli(cfg.alpha0, cfg.beta0)
        )

        # Number of run-length states (R = max_run_length + 1)
        self.R: int = int(cfg.max_run_length) + 1

        # Run-length posterior at previous step
        self.R_prev: ArrayF = np.zeros(self.R, dtype=float)
        self.R_prev[0] = 1.0

        # Conjugate likelihood state for each run length.
        self.lik: ConjugateLikelihood = self._likelihood_template.clone()
        self.lik.init_stats(self.R)

        # Bookkeeping
        self.t: int = 0
        self.normalization_issues_: int = 0  # count of rescale/fallback events

    # Expose internal Beta parameters for compatibility with existing tests
    @property
    def alpha(self) -> ArrayF:
        if not hasattr(self.lik, "stats") or not hasattr(self.lik.stats, "alpha"):
            raise AttributeError("alpha is only available for BetaBernoulli likelihoods.")
        return self.lik.stats.alpha  # type: ignore[union-attr]

    @property
    def beta(self) -> ArrayF:
        if not hasattr(self.lik, "stats") or not hasattr(self.lik.stats, "beta"):
            raise AttributeError("beta is only available for BetaBernoulli likelihoods.")
        return self.lik.stats.beta  # type: ignore[union-attr]

    # ----------------------- Public API -----------------------

    def reset(self) -> None:
        """Reset the filter to its prior state."""
        self.R_prev.fill(0.0)
        self.R_prev[0] = 1.0
        self.lik = self._likelihood_template.clone()
        self.lik.init_stats(self.R)
        self.t = 0
        self.normalization_issues_ = 0

    @staticmethod
    def _is_missing_observation(x_t: Any) -> bool:
        if x_t is None:
            return True
        try:
            arr = np.asarray(x_t, dtype=float)
        except (TypeError, ValueError):
            return False
        return bool(arr.size > 0 and np.isnan(arr).all())

    def update(self, x_t: Any) -> Dict[str, float]:
        """
        Ingest a single observation and update the run-length posterior online.

        Parameters
        ----------
        x_t : Any
            Observation at time t. The configured likelihood defines valid
            non-missing values. ``None`` and all-NaN numeric values are treated
            as missing observations: the run-length transition advances, but
            likelihood sufficient statistics are not updated with data.

        Returns
        -------
        dict with keys:
            "cp_prob"      : P(r_t=0 | x_{1:t})
            "map_run_length": argmax run-length at time t
            "pred_mean"    : P(x_t=1 | x_{1:t-1}) (mixture, before consuming x_t)
            "log_evidence" : log normalizer before pruning/top-k approximation
            "approximation_error": removed mass fraction from approximations
        """
        missing = self._is_missing_observation(x_t)

        # (1) One-step-ahead predictive mean BEFORE consuming x_t
        state_means: ArrayF = self.lik.predictive_mean()        # shape (R,)
        mixture_mean = np.asarray(np.dot(self.R_prev, state_means), dtype=float)
        pred_mean = float(mixture_mean.reshape(-1)[0])

        # (2) Per-state predictive probability for the realized x_t
        pred: ArrayF = (
            np.ones(self.R, dtype=float)
            if missing
            else self.lik.predictive_prob(x_t)
        )             # shape (R,)

        # (3) Hazard per state at time t
        H = np.fromiter(
            (self.hazard.prob(r, self.t) for r in range(self.R)),
            count=self.R,
            dtype=float,
        )
        one_m_H = 1.0 - H

        # (4) BOCPD recursion (unnormalized): growth and changepoint mass
        R_next = np.zeros_like(self.R_prev)
        if self.R > 1:
            R_next[1:] = self.R_prev[:-1] * one_m_H[:-1] * pred[:-1]  # growth
        dropped_growth = float(self.R_prev[-1] * one_m_H[-1] * pred[-1])

        # Changepoint probability: sum hazard mass and use the fresh-segment
        # prior predictive, not the posterior predictive of the previous r=0.
        # cp_scale is retained only for compatibility and is not canonical BOCPD.
        cp_mass = float(np.dot(self.R_prev, H))
        prior_predictive = 1.0 if missing else float(self.lik.prior_predictive_prob(x_t))
        R_next[0] = cp_mass * prior_predictive
        R_next[0] *= float(self.cfg.cp_scale)

        # (5) Normalize robustly (underflow guard, tail pruning, optional top-k)
        tracked_normalizer = float(R_next.sum())
        full_normalizer = tracked_normalizer + dropped_growth
        log_evidence = (
            float(np.log(full_normalizer))
            if np.isfinite(full_normalizer) and full_normalizer > 0.0
            else float("-inf")
        )
        truncation_error = (
            dropped_growth / full_normalizer
            if np.isfinite(full_normalizer) and full_normalizer > 0.0
            else 0.0
        )
        pruning_error = self._prune_and_normalize(R_next)
        approximation_error = truncation_error + (1.0 - truncation_error) * pruning_error

        # (6) Update sufficient statistics for the likelihood
        #     Order matters: grow old segments first, then reset r=0.
        if missing:
            self.lik.update_growth_missing()
            self.lik.update_cp_missing()
        else:
            self.lik.update_growth(x_t)   # shift r -> r+1 and add x_t
            self.lik.update_cp(x_t)       # set r=0 with prior + x_t

        # (7) Commit
        self.R_prev = R_next
        self.t += 1

        return {
            "cp_prob": float(R_next[0]),
            "map_run_length": int(np.argmax(R_next)),
            "pred_mean": pred_mean,
            "log_evidence": log_evidence,
            "approximation_error": float(np.clip(approximation_error, 0.0, 1.0)),
        }

    def run(self, x: Sequence[Any]) -> BOCPDResult:
        """
        Process a full sequence in one pass (still online internally).

        Returns
        -------
        BOCPDResult
        """
        self.reset()
        return self.update_many(x)

    def update_many(self, x: Sequence[Any]) -> BOCPDResult:
        """Process a batch without resetting existing online state."""
        T = len(x)

        cp_probs = np.empty(T, dtype=float)
        map_r = np.empty(T, dtype=np.int_)
        pred_means = np.empty(T, dtype=float)
        log_evidence = np.empty(T, dtype=float)
        approximation_error = np.empty(T, dtype=float)
        rl_store = np.empty((T, self.R), dtype=float) if self.cfg.store_run_length_posterior else None

        for t, val in enumerate(x):
            out = self.update(val)
            cp_probs[t] = out["cp_prob"]
            map_r[t] = out["map_run_length"]
            pred_means[t] = out["pred_mean"]
            log_evidence[t] = out["log_evidence"]
            approximation_error[t] = out["approximation_error"]
            if rl_store is not None:
                rl_store[t, :] = self.R_prev

        return BOCPDResult(
            cp_prob=cp_probs,
            map_run_length=map_r,
            pred_mean=pred_means,
            run_length_posterior=rl_store,
            log_evidence=log_evidence,
            approximation_error=approximation_error,
            diagnostics={
                "normalization_issues": self.normalization_issues_,
                "posterior_is_calibrated": self.cfg.cp_scale == 1.0,
                "cp_scale": self.cfg.cp_scale,
                "prune_epsilon": self.cfg.prune_epsilon,
                "top_k": self.cfg.top_k,
                "max_run_length": self.cfg.max_run_length,
                "likelihood": type(self.lik).__name__,
            },
        )

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible checkpoint for the current online state."""
        return {
            "kind": "BOCPD",
            "t": self.t,
            "R": self.R,
            "R_prev": self.R_prev.tolist(),
            "normalization_issues": self.normalization_issues_,
            "cfg": {
                "max_run_length": self.cfg.max_run_length,
                "prune_epsilon": self.cfg.prune_epsilon,
                "prune_relative": self.cfg.prune_relative,
                "top_k": self.cfg.top_k,
                "cp_scale": self.cfg.cp_scale,
            },
            "likelihood": self.lik.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore an online state produced by :meth:`state_dict`."""
        if state.get("kind") != "BOCPD":
            raise ValueError("state kind does not match BOCPD.")
        if int(state["R"]) != self.R:
            raise ValueError("state run-length support does not match this model.")
        self.t = int(state["t"])
        self.R_prev = np.asarray(state["R_prev"], dtype=float)
        if self.R_prev.shape != (self.R,):
            raise ValueError("state R_prev shape does not match this model.")
        total = float(self.R_prev.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("state R_prev must contain positive finite mass.")
        self.R_prev /= total
        self.normalization_issues_ = int(state.get("normalization_issues", 0))
        self.lik.load_state_dict(state["likelihood"])

    # ------------------- Numerics & pruning -------------------

    def _prune_and_normalize(self, R_next: ArrayF) -> float:
        """
        In-place stabilization: rescale if needed, prune tiny mass, optional top-K,
        and renormalize. Protects against underflow in long, low-hazard stretches.
        """
        eps = float(self.cfg.stabilizer)
        removed_mass = 0.0

        # Rescale if sum underflows
        total = float(R_next.sum())
        if not np.isfinite(total) or total <= eps:
            m = float(R_next.max(initial=0.0))
            if np.isfinite(m) and m > eps:
                R_next /= m
                total = float(R_next.sum())
            else:
                # Catastrophic underflow: keep mass at r=0 to avoid spurious spikes
                R_next.fill(0.0)
                R_next[0] = 1.0
                self.normalization_issues_ += 1
                return 1.0

        # First normalize to probability scale
        R_next /= total

        # Tail pruning (relative to max by default)
        pe = float(self.cfg.prune_epsilon)
        if pe > 0.0:
            thr = pe * float(R_next.max()) if self.cfg.prune_relative else pe
            if thr > 0.0:
                prune_mask = R_next < thr
                removed_mass += float(R_next[prune_mask].sum())
                R_next[prune_mask] = 0.0
                s = float(R_next.sum())
                if s <= eps:
                    # If everything got pruned, keep the argmax
                    i = int(np.argmax(R_next))
                    R_next.fill(0.0)
                    R_next[i] = 1.0
                    self.normalization_issues_ += 1
                else:
                    R_next /= s

        # Optional top-K pruning (keep the K largest states + r=0)
        if self.cfg.top_k is not None:
            k = int(self.cfg.top_k)
            if 0 < k < R_next.size:
                keep = np.argpartition(R_next, -k)[-k:]
                mask = np.zeros_like(R_next, dtype=bool)
                mask[keep] = True
                mask[0] = True  # always preserve r=0
                removed_mass += float(R_next[~mask].sum())
                R_next[~mask] = 0.0
                s = float(R_next.sum())
                if s > eps and np.isfinite(s):
                    R_next /= s
                else:
                    # Degenerate after pruning → keep MAP state
                    i = int(np.argmax(R_next))
                    R_next.fill(0.0)
                    R_next[i] = 1.0
                    self.normalization_issues_ += 1
        return float(np.clip(removed_mass, 0.0, 1.0))


def extract_changepoint_alerts(
    result: BOCPDResult,
    config: BOCPDAlertConfig | None = None,
) -> NDArray[np.int_]:
    """Extract changepoint alert indices from a BOCPD posterior result."""
    cfg = config or BOCPDAlertConfig()
    cp_prob = np.asarray(result.cp_prob, dtype=float)
    if cp_prob.size == 0:
        return np.array([], dtype=int)

    candidates = np.ones(cp_prob.size, dtype=bool)
    if cfg.probability_threshold is None:
        candidates.fill(False)
    else:
        candidates &= cp_prob >= cfg.probability_threshold

    if cfg.require_local_peak:
        peaks = np.ones(cp_prob.size, dtype=bool)
        if cp_prob.size > 1:
            peaks[0] = cp_prob[0] >= cp_prob[1]
            peaks[-1] = cp_prob[-1] > cp_prob[-2]
        if cp_prob.size > 2:
            peaks[1:-1] = (cp_prob[1:-1] >= cp_prob[:-2]) & (
                cp_prob[1:-1] > cp_prob[2:]
            )
        candidates &= peaks

    if cfg.use_run_length_reset:
        resets = np.zeros(cp_prob.size, dtype=bool)
        if cp_prob.size > 1:
            run_lengths = np.asarray(result.map_run_length, dtype=int)
            resets[1:] = run_lengths[1:] < run_lengths[:-1]
        if cfg.probability_threshold is None:
            candidates = resets
        else:
            candidates &= resets

    raw = np.flatnonzero(candidates)
    if cfg.min_spacing <= 1 or raw.size <= 1:
        return raw.astype(int)

    kept: list[int] = []
    last = -cfg.min_spacing
    for idx in raw.tolist():
        if idx - last >= cfg.min_spacing:
            kept.append(int(idx))
            last = int(idx)
    return np.asarray(kept, dtype=int)
