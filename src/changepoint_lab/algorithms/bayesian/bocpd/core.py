# bocpd.py
# MIT License
# (c) 2025

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Set

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
    """
    alpha0: float = 1.0
    beta0: float = 1.0
    max_run_length: int = 512
    store_run_length_posterior: bool = True
    prune_epsilon: float = 1e-6
    prune_relative: bool = True
    stabilizer: float = 1e-300
    top_k: Optional[int] = None
    cp_scale: float = 20.0

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
    """
    cp_prob: ArrayF
    map_run_length: NDArray[np.int_]
    pred_mean: ArrayF
    run_length_posterior: Optional[ArrayF]

    def __post_init__(self) -> None:
        T = len(self.cp_prob)
        if len(self.map_run_length) != T or len(self.pred_mean) != T:
            raise ValueError("cp_prob, map_run_length and pred_mean must have equal length")
        if self.run_length_posterior is not None and self.run_length_posterior.shape[0] != T:
            raise ValueError("run_length_posterior must have first dimension == len(cp_prob)")


# ------------------------------ Model ---------------------------------

class BOCPD:
    """
    Bayesian Online Changepoint Detection (Adams & MacKay, 2007) for Bernoulli {0,1} data
    with a pluggable conjugate likelihood (here: Beta–Bernoulli).

    Hazard H(r, t) controls the prior CP probability; use ConstantHazard(λ) for the classic model.
    """

    def __init__(self, hazard: Hazard, cfg: BOCPDConfig = BOCPDConfig()) -> None:
        self.hazard = hazard
        self.cfg = cfg

        # Number of run-length states (R = max_run_length + 1)
        self.R: int = int(cfg.max_run_length) + 1

        # Run-length posterior at previous step
        self.R_prev: ArrayF = np.zeros(self.R, dtype=float)
        self.R_prev[0] = 1.0

        # Conjugate likelihood (Beta–Bernoulli)
        self.lik: ConjugateLikelihood = BetaBernoulli(cfg.alpha0, cfg.beta0)
        self.lik.init_stats(self.R)

        # Bookkeeping
        self.t: int = 0
        self.normalization_issues_: int = 0  # count of rescale/fallback events

    # Expose internal Beta parameters for compatibility with existing tests
    @property
    def alpha(self) -> ArrayF:
        return self.lik.stats.alpha  # type: ignore[union-attr]

    @property
    def beta(self) -> ArrayF:
        return self.lik.stats.beta  # type: ignore[union-attr]

    # ----------------------- Public API -----------------------

    def reset(self) -> None:
        """Reset the filter to its prior state."""
        self.R_prev.fill(0.0)
        self.R_prev[0] = 1.0
        self.lik = BetaBernoulli(self.cfg.alpha0, self.cfg.beta0)
        self.lik.init_stats(self.R)
        self.t = 0
        self.normalization_issues_ = 0

    def update(self, x_t: int | bool) -> Dict[str, float]:
        """
        Ingest a single observation and update the run-length posterior online.

        Parameters
        ----------
        x_t : int|bool
            Observation at time t (0/1).

        Returns
        -------
        dict with keys:
            "cp_prob"      : P(r_t=0 | x_{1:t})
            "map_run_length": argmax run-length at time t
            "pred_mean"    : P(x_t=1 | x_{1:t-1}) (mixture, before consuming x_t)
        """
        xi = bool(x_t)

        # (1) One-step-ahead predictive mean BEFORE consuming x_t
        state_means: ArrayF = self.lik.predictive_mean()        # shape (R,)
        pred_mean = float(np.dot(self.R_prev, state_means))     # mixture

        # (2) Per-state predictive probability for the realized x_t
        pred: ArrayF = self.lik.predictive_prob(xi)             # shape (R,)

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
        # Changepoint probability: sum hazard mass and use prior predictive for r=0
        cp_mass = float(np.dot(self.R_prev, H))
        R_next[0] = cp_mass * float(pred[0]) * float(self.cfg.cp_scale)

        # (5) Normalize robustly (underflow guard, tail pruning, optional top-k)
        self._prune_and_normalize(R_next)

        # (6) Update sufficient statistics for the likelihood
        #     Order matters: grow old segments first, then reset r=0.
        self.lik.update_growth(xi)   # shift r -> r+1 and add x_t
        self.lik.update_cp(xi)       # set r=0 with prior + x_t

        # (7) Commit
        self.R_prev = R_next
        self.t += 1

        return {
            "cp_prob": float(R_next[0]),
            "map_run_length": int(np.argmax(R_next)),
            "pred_mean": pred_mean,
        }

    def run(self, x: Sequence[int | bool]) -> BOCPDResult:
        """
        Process a full sequence in one pass (still online internally).

        Returns
        -------
        BOCPDResult
        """
        self.reset()
        T = len(x)

        cp_probs = np.empty(T, dtype=float)
        map_r = np.empty(T, dtype=np.int_)
        pred_means = np.empty(T, dtype=float)
        rl_store = np.empty((T, self.R), dtype=float) if self.cfg.store_run_length_posterior else None

        for t, val in enumerate(x):
            out = self.update(val)
            cp_probs[t] = out["cp_prob"]
            map_r[t] = out["map_run_length"]
            pred_means[t] = out["pred_mean"]
            if rl_store is not None:
                rl_store[t, :] = self.R_prev

        return BOCPDResult(
            cp_prob=cp_probs,
            map_run_length=map_r,
            pred_mean=pred_means,
            run_length_posterior=rl_store,
        )

    # ------------------- Numerics & pruning -------------------

    def _prune_and_normalize(self, R_next: ArrayF) -> None:
        """
        In-place stabilization: rescale if needed, prune tiny mass, optional top-K,
        and renormalize. Protects against underflow in long, low-hazard stretches.
        """
        eps = float(self.cfg.stabilizer)

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
                return

        # First normalize to probability scale
        R_next /= total

        # Tail pruning (relative to max by default)
        pe = float(self.cfg.prune_epsilon)
        if pe > 0.0:
            thr = pe * float(R_next.max()) if self.cfg.prune_relative else pe
            if thr > 0.0:
                R_next[R_next < thr] = 0.0
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
