# bocpd.py
# MIT License
# (c) 2025

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.floating]
ArrayB = NDArray[np.bool_]


# ----------------------------- Hazards ---------------------------------

class Hazard:
    """Abstract hazard: P(changepoint at t | run_length=r, t)."""

    def prob(self, r: int, t: int) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantHazard(Hazard):
    """
    Constant hazard: h = 1 / mean_run_length.

    Parameters
    ----------
    mean_run_length : float
        Expected segment length under the prior.
    eps : float
        Numerical clamp for extreme values.
    """
    mean_run_length: float
    eps: float = 1e-12

    def prob(self, r: int, t: int) -> float:
        h = 1.0 / float(self.mean_run_length)
        # clip for safety
        return float(np.clip(h, self.eps, 1.0 - self.eps))


@dataclass(frozen=True)
class ScheduledHazard(Hazard):
    """
    Time-of-day (or periodic) scheduled hazard.

    Parameters
    ----------
    schedule : Sequence[float]
        Hazard values for indices 0..N-1. Will be clipped to (eps, 1-eps).
    period : int
        Period used as t % period to index into schedule. Must equal len(schedule).
    eps : float
        Numerical clamp for extreme values.
    """
    schedule: Sequence[float]
    period: int
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if len(self.schedule) != self.period:
            raise ValueError("len(schedule) must equal period.")
        if not (0 < self.eps < 0.5):
            raise ValueError("eps must be in (0, 0.5).")

    def prob(self, r: int, t: int) -> float:
        h = float(self.schedule[t % self.period])
        return float(np.clip(h, self.eps, 1.0 - self.eps))


@dataclass(frozen=True)
class BoostedBoundaryHazard(Hazard):
    """
    Wrapper that boosts a base hazard at specific boundary indices (e.g., t % N == 0).

    Parameters
    ----------
    base : Hazard
        Base hazard to evaluate first.
    period : int
        Period of boundary check (t % period).
    boundary_indices : set[int]
        Indices in {0..period-1} where the boost applies.
    boost_factor : float
        Multiplier applied to base hazard at the boundary. Final hazard is clipped to (eps, 1-eps).
    eps : float
        Numerical clamp.
    """
    base: Hazard
    period: int
    boundary_indices: frozenset[int]
    boost_factor: float = 10.0
    eps: float = 1e-12

    def prob(self, r: int, t: int) -> float:
        h = self.base.prob(r, t)
        if (t % self.period) in self.boundary_indices:
            h = h * self.boost_factor
        return float(np.clip(h, self.eps, 1.0 - self.eps))


# ----------------------------- BOCPD core ------------------------------

@dataclass(frozen=True)
class BOCPDConfig:
    """
    Configuration for Beta–Bernoulli BOCPD.

    Attributes
    ----------
    alpha0, beta0 : float
        Beta prior hyperparameters (alpha > 0, beta > 0).
    max_run_length : int
        Truncation for the run-length support r ∈ {0,...,max_run_length}.
        Complexity is O(T * max_run_length).
    store_run_length_posterior : bool
        If True, we store the full run-length posterior over time (for heatmaps).
    """
    alpha0: float = 1.0
    beta0: float = 1.0
    max_run_length: int = 512
    store_run_length_posterior: bool = True
    # Numerical robustness knobs
    prune_epsilon: float = 1e-6
    prune_relative: bool = True
    stabilizer: float = 1e-300


@dataclass
class BOCPDResult:
    """
    Results from processing a sequence.

    Attributes
    ----------
    cp_prob : ArrayF, shape (T,)
        P(r_t = 0 | x_{1:t}) at each time.
    map_run_length : NDArray[np.int64], shape (T,)
        Argmax run-length at each time t.
    pred_mean : ArrayF, shape (T,)
        One-step-ahead predictive mean P(x_t=1 | x_{1:t-1}) before seeing x_t.
    run_length_posterior : Optional[ArrayF], shape (T, R+1)
        Posterior over run-length if requested.
    """
    cp_prob: ArrayF
    map_run_length: NDArray[np.int64]
    pred_mean: ArrayF
    run_length_posterior: Optional[ArrayF]


class BOCPD:
    """
    Bayesian Online Changepoint Detection (Adams & MacKay, 2007) for Bernoulli {0,1} data
    with Beta(alpha0, beta0) prior (conjugate updates).

    Hazard H(r, t) controls the prior CP probability; use ConstantHazard(λ) for the classic model.
    """

    def __init__(self, hazard: Hazard, cfg: BOCPDConfig = BOCPDConfig()) -> None:
        if cfg.alpha0 <= 0 or cfg.beta0 <= 0:
            raise ValueError("alpha0, beta0 must be > 0.")
        if cfg.max_run_length < 1:
            raise ValueError("max_run_length must be >= 1.")
        self.hazard = hazard
        self.cfg = cfg

        # Internal state (set in reset())
        self.R_prev: ArrayF  # run-length posterior at t-1
        self.alpha: ArrayF   # Beta alpha for each run-length state (after t-1)
        self.beta: ArrayF    # Beta beta for each run-length state (after t-1)
        self.t: int          # time index (processed points)
        self.normalization_issues_: int = 0  # count of underflow rescales/fallbacks

        self.reset()

    def reset(self) -> None:
        """Reset the filter to its prior state."""
        R = self.cfg.max_run_length
        self.R_prev = np.zeros(R + 1, dtype=float)
        self.R_prev[0] = 1.0  # At t=0, run-length=0 with prob 1
        self.alpha = np.full(R + 1, self.cfg.alpha0, dtype=float)
        self.beta = np.full(R + 1, self.cfg.beta0, dtype=float)
        self.t = 0
        self.normalization_issues_ = 0

    # ---- predictive helpers ----

    def _predictive_mass(self, x_t: int) -> ArrayF:
        """
        Predictive likelihood p(x_t | state r) for all run-length states r at time t,
        using the Beta posterior parameters from t-1.
        """
        a = self.alpha
        b = self.beta
        # Bernoulli predictive: P(1) = a/(a+b), P(0) = b/(a+b)
        denom = a + b
        with np.errstate(divide="ignore", invalid="ignore"):
            if x_t == 1:
                pred = a / denom
            else:
                pred = b / denom
        pred = np.nan_to_num(pred, nan=0.5)  # in case of degenerate (shouldn't happen with a,b>0)
        return pred

    def _mixture_predictive_mean(self) -> float:
        """
        One-step predictive mean P(x_t=1 | x_{1:t-1}),
        mixing over run-length states with R_prev.
        """
        a = self.alpha
        b = self.beta
        p1 = a / (a + b)
        return float(np.dot(self.R_prev, p1))
    
    def _prune_and_normalize(self, R_next: ArrayF) -> None:
        """In-place stabilization: rescale if needed, prune tiny mass, and renormalize.

        Protects against underflow when hazards are very small over long stretches.
        Mirrors tail pruning (Adams & MacKay, 2007, Sec. 2.4) without changing the API.
        """
        eps = float(self.cfg.stabilizer)
        total = float(R_next.sum())
        if not np.isfinite(total) or total <= eps:
            # Max-rescale (soft log-sum-exp trick)
            m = float(R_next.max(initial=0.0))
            if np.isfinite(m) and m > eps:
                R_next /= m
                total = float(R_next.sum())
            else:
                # Catastrophic underflow: keep r=0 to avoid spurious CP spikes
                R_next.fill(0.0)
                R_next[0] = 1.0
                self.normalization_issues_ += 1
                return

        # Normalize onto probability scale before pruning
        R_next /= total

        # Tail pruning (relative by default)
        pe = float(self.cfg.prune_epsilon)
        if pe > 0.0:
            thr = pe * float(R_next.max()) if self.cfg.prune_relative else pe
            if thr > 0.0:
                R_next[R_next < thr] = 0.0
                total2 = float(R_next.sum())
                if total2 <= eps:
                    imax = int(np.argmax(R_next))
                    R_next.fill(0.0)
                    R_next[imax] = 1.0
                    self.normalization_issues_ += 1
                else:
                    R_next /= total2


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
            "cp_prob" : P(r_t=0 | x_{1:t})
            "map_run_length" : argmax run-length at time t
            "pred_mean" : P(x_t=1 | x_{1:t-1})
        """
        xt = int(bool(x_t))  # normalize to {0,1}
        R = self.cfg.max_run_length

        # Predictive mixture before seeing x_t
        pred_mean = self._mixture_predictive_mean()

        # Predictive by state
        pred = self._predictive_mass(xt)

        # Hazard per state at time t
        H = np.fromiter((self.hazard.prob(r, self.t) for r in range(R + 1)), count=R + 1, dtype=float)
        one_m_H = 1.0 - H

        # Growth: r -> r+1
        growth = self.R_prev[:-1] * one_m_H[:-1] * pred[:-1]

        # Changepoint: r -> 0
        cp_mass = np.sum(self.R_prev * H * pred)
        R_new = np.empty_like(self.R_prev)
        R_new[0] = cp_mass
        R_new[1:] = growth

        # Robust normalization with tail pruning
        self._prune_and_normalize(R_new)

        # Update Beta parameters for next step
        alpha_new = np.empty_like(self.alpha)
        beta_new = np.empty_like(self.beta)
        # Run-length 0: new segment starts with prior then observe x_t
        alpha_new[0] = self.cfg.alpha0 + xt
        beta_new[0] = self.cfg.beta0 + (1 - xt)
        # Growth: r -> r+1 (carry sufficient stats and update with x_t)
        alpha_new[1:] = self.alpha[:-1] + xt
        beta_new[1:] = self.beta[:-1] + (1 - xt)

        # Commit
        self.R_prev = R_new
        self.alpha = alpha_new
        self.beta = beta_new
        self.t += 1

        # Diagnostics
        cp_prob = float(R_new[0])
        map_r = int(np.argmax(R_new))

        return {"cp_prob": cp_prob, "map_run_length": map_r, "pred_mean": pred_mean}

    def run(self, x: Sequence[int | bool]) -> BOCPDResult:
        """
        Process a full sequence in one pass (still online internally).

        Returns
        -------
        BOCPDResult
        """
        self.reset()
        T = len(x)
        R = self.cfg.max_run_length
        cp_probs = np.empty(T, dtype=float)
        map_r = np.empty(T, dtype=np.int64)
        pred_means = np.empty(T, dtype=float)
        R_store = np.empty((T, R + 1), dtype=float) if self.cfg.store_run_length_posterior else None

        for t, val in enumerate(x):
            res = self.update(val)
            cp_probs[t] = res["cp_prob"]
            map_r[t] = res["map_run_length"]
            pred_means[t] = res["pred_mean"]
            if R_store is not None:
                R_store[t, :] = self.R_prev

        return BOCPDResult(
            cp_prob=cp_probs,
            map_run_length=map_r,
            pred_mean=pred_means,
            run_length_posterior=R_store,
        )
