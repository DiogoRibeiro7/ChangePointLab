# likelihoods.py
# MIT License
# (c) 2025
"""
Conjugate-exponential family likelihoods for BOCPD.

Design goals
------------
- Keep BOCPD recursion unchanged; swap just the likelihood "module".
- Store sufficient statistics for every run-length state (size R = Rmax+1).
- Provide vectorized predictive probabilities and in-place roll-forward updates.

This file defines:
- ConjugateLikelihood (abstract interface)
- BetaBernoulli (fully implemented)
- PoissonGamma (skeleton)
- GaussianNIW (skeleton)

Notes
-----
- Shapes: all per-state arrays are 1D of length R (Rmax+1).
- BOCPD calls:
    1) predictive_prob(x_t, stats) -> (R,) predictive probabilities
    2) update_cp(x_t, stats) -> write r=0 stats (reset segment with x_t)
    3) update_growth(x_t, stats) -> write r=1..R-1 stats (growth of segment)
- For performance, all methods are vectorized and avoid Python loops.

References
----------
- Adams & MacKay (2007): Bayesian Online Changepoint Detection (arXiv:0710.3742)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Protocol

import numpy as np
from numpy.typing import NDArray


# ------------------------- Common type aliases --------------------------

ArrayF = NDArray[np.floating]
ArrayI = NDArray[np.integer]
ArrayB = NDArray[np.bool_]


# --------------------------- Base interface -----------------------------

class ConjugateLikelihood(ABC):
    """
    Abstract interface for conjugate likelihood models in BOCPD.

    A concrete implementation must manage its per-state sufficient statistics and
    provide vectorized predictive probabilities and roll-forward updates.
    """

    @abstractmethod
    def init_stats(self, R: int) -> None:
        """
        Allocate and initialize per-state sufficient statistics for R states.

        Parameters
        ----------
        R : int
            Number of run-length states (R = Rmax + 1).
        """
        ...

    @abstractmethod
    def predictive_prob(self, x_t, /) -> ArrayF:
        """
        Compute p(x_t | state parameters) for *each* run-length state.

        Returns
        -------
        probs : ArrayF, shape (R,)
            Predictive probability (not log) for each state.
        """
        ...

    @abstractmethod
    def prior_predictive_prob(self, x_t, /) -> float:
        """Compute p(x_t) under the fresh segment prior."""
        ...

    @abstractmethod
    def update_cp(self, x_t) -> None:
        """
        Reset-update for r = 0 given observation x_t.
        Writes stats for the new segment of length 1.
        """
        ...

    @abstractmethod
    def update_growth(self, x_t) -> None:
        """
        Growth-update for r = 1..R-1 given observation x_t.
        Shifts stats from r -> r+1 and applies the sufficient-stat increment.
        """
        ...

    @abstractmethod
    def predictive_mean(self) -> ArrayF:
        """
        Optional convenience: posterior predictive mean for each state.

        Returns
        -------
        mean : ArrayF, shape (R,)
        """
        ...


# ============================= Beta–Bernoulli ===========================

@dataclass
class BetaBernoulliStats:
    """Sufficient stats per run-length state."""
    alpha: ArrayF  # successes + alpha0
    beta: ArrayF   # failures  + beta0


class BetaBernoulli(ConjugateLikelihood):
    """
    Conjugate Bernoulli with Beta prior.

    Parameters
    ----------
    alpha0 : float
        Prior alpha > 0.
    beta0 : float
        Prior beta  > 0.

    Notes
    -----
    Predictive:
        P(x=1 | α,β) = α/(α+β);  P(x=0 | α,β) = β/(α+β)
    Updates (for a single new Bernoulli x in {0,1}):
        α' = α + x,   β' = β + (1 - x)
    """

    def __init__(self, alpha0: float = 1.0, beta0: float = 1.0) -> None:
        if not (alpha0 > 0.0 and beta0 > 0.0):
            raise ValueError("alpha0 and beta0 must be > 0.")
        self.alpha0: float = float(alpha0)
        self.beta0: float = float(beta0)
        self.stats: Optional[BetaBernoulliStats] = None  # filled by init_stats()

    # ------------------------ Interface methods ------------------------

    def init_stats(self, R: int) -> None:
        if R <= 0:
            raise ValueError("R must be >= 1 (R = Rmax + 1).")
        alpha = np.full(R, self.alpha0, dtype=float)
        beta = np.full(R, self.beta0, dtype=float)
        self.stats = BetaBernoulliStats(alpha=alpha, beta=beta)

    def predictive_prob(self, x_t, /) -> ArrayF:
        """
        Vectorized predictive probability for observation x_t in {0,1,True,False}.
        """
        if self.stats is None:
            raise RuntimeError("Call init_stats(R) before predictive_prob().")

        # Coerce to {0,1}
        xi = 1.0 if bool(x_t) else 0.0
        α = self.stats.alpha
        β = self.stats.beta
        denom = α + β
        # Safe division: denom>0 because alpha0,beta0>0 and updates are nonnegative
        p1 = α / denom  # P(x=1)
        # Return P(x=xi) across all states
        return p1 if xi == 1.0 else (1.0 - p1)

    def prior_predictive_prob(self, x_t, /) -> float:
        """Predictive probability under the initial Beta prior."""
        xi = 1.0 if bool(x_t) else 0.0
        p1 = self.alpha0 / (self.alpha0 + self.beta0)
        return float(p1 if xi == 1.0 else (1.0 - p1))

    def predictive_mean(self) -> ArrayF:
        if self.stats is None:
            raise RuntimeError("Call init_stats(R) before predictive_mean().")
        α, β = self.stats.alpha, self.stats.beta
        return α / (α + β)

    def update_cp(self, x_t) -> None:
        """
        Reset (r=0): incorporate x_t into a fresh segment.
        """
        if self.stats is None:
            raise RuntimeError("Call init_stats(R) before update_cp().")
        xi = 1.0 if bool(x_t) else 0.0
        # r=0 is always "fresh prior + x_t"
        self.stats.alpha[0] = self.alpha0 + xi
        self.stats.beta[0] = self.beta0 + (1.0 - xi)

    def update_growth(self, x_t) -> None:
        """
        Growth (r -> r+1 for r>=0): shift stats and add x_t to sufficient stats.

        After this call:
            stats[:, 1:] reflect grown segments that just included x_t.
            stats[:, 0] should be set separately by update_cp(x_t).
        """
        if self.stats is None:
            raise RuntimeError("Call init_stats(R) before update_growth().")
        xi = 1.0 if bool(x_t) else 0.0

        α, β = self.stats.alpha, self.stats.beta
        # Shift right for growth transitions: r -> r+1
        # Use roll with overwrite to avoid extra allocations.
        α[1:] = α[:-1]
        β[1:] = β[:-1]
        # Add observation to all grown states (r>=1)
        α[1:] += xi
        β[1:] += (1.0 - xi)
        # Note: α[0], β[0] are left for update_cp(x_t) to set.


# ============================== Poisson–Gamma ===========================

@dataclass
class PoissonGammaStats:
    """
    Sufficient stats per state for Poisson with Gamma prior on rate λ.

    We track (shape, rate) in Gamma(shape, rate) parameterization.
    Posterior after n counts with sum s is Gamma(shape0 + s, rate0 + n).

    For full implementation you will likely also track per-state accumulated
    event counts if you prefer to maintain as running sufficient statistics.
    """
    shape: ArrayF
    rate: ArrayF


class PoissonGamma(ConjugateLikelihood):
    """
    Conjugate Poisson with Gamma prior.

    Parameters
    ----------
    shape0 : float
        Prior shape > 0.
    rate0 : float
        Prior rate  > 0.

    Predictive (one-step):
        x ~ NegBinomial-like marginal. For integer x>=0, the closed form is:
        P(x|shape,rate) = comb(x+shape-1, x) * (rate/(1+rate))**shape * (1/(1+rate))**x
        when parameterized appropriately. See standard Poisson-Gamma predictive.
    """

    def __init__(self, shape0: float = 1.0, rate0: float = 1.0) -> None:
        if not (shape0 > 0.0 and rate0 > 0.0):
            raise ValueError("shape0 and rate0 must be > 0.")
        self.shape0 = float(shape0)
        self.rate0 = float(rate0)
        self.stats: Optional[PoissonGammaStats] = None

    def init_stats(self, R: int) -> None:
        if R <= 0:
            raise ValueError("R must be >= 1 (R = Rmax + 1).")
        self.stats = PoissonGammaStats(
            shape=np.full(R, self.shape0, dtype=float),
            rate=np.full(R, self.rate0, dtype=float),
        )

    def predictive_prob(self, x_t, /) -> ArrayF:
        if self.stats is None:
            raise RuntimeError("Call init_stats(R) before predictive_prob().")
        # TODO: Implement the closed-form Poisson–Gamma predictive pmf for integer x_t.
        # For now, raise to make the placeholder explicit.
        raise NotImplementedError("PoissonGamma.predictive_prob is not yet implemented.")

    def prior_predictive_prob(self, x_t, /) -> float:
        raise NotImplementedError("PoissonGamma.prior_predictive_prob is not yet implemented.")

    def predictive_mean(self) -> ArrayF:
        if self.stats is None:
            raise RuntimeError("Call init_stats(R) before predictive_mean().")
        # E[λ] = shape/rate
        return self.stats.shape / self.stats.rate

    def update_cp(self, x_t) -> None:
        if self.stats is None:
            raise RuntimeError("Call init_stats(R) before update_cp().")
        # TODO: For integer count x_t >= 0:
        #   shape[0] = shape0 + x_t
        #   rate[0]  = rate0  + 1
        raise NotImplementedError("PoissonGamma.update_cp is not yet implemented.")

    def update_growth(self, x_t) -> None:
        if self.stats is None:
            raise RuntimeError("Call init_stats(R) before update_growth().")
        # TODO:
        #   shape[1:] = shape[:-1] + x_t
        #   rate[1:]  = rate[:-1]  + 1
        raise NotImplementedError("PoissonGamma.update_growth is not yet implemented.")


# ============================== Gaussian–NIW ============================

@dataclass
class GaussianNIWStats:
    """
    Sufficient stats per state for multivariate Normal with NIW prior.

    Each state r holds an NIW prior/posterior: (m, kappa, nu, Psi)
      - m    : mean vector (d,)
      - kappa: scalar > 0
      - nu   : df > d - 1
      - Psi  : scale matrix (d,d), SPD

    For an efficient implementation, you can keep:
      - sum of x, sum of x x^T (per state) OR directly maintain NIW params.
      - predictive is multivariate Student-t.
    """
    # Minimal placeholders (not allocated until you know d)
    m: Optional[ArrayF] = None     # shape (R, d)
    kappa: Optional[ArrayF] = None # shape (R,)
    nu: Optional[ArrayF] = None    # shape (R,)
    Psi: Optional[ArrayF] = None   # shape (R, d, d)


class GaussianNIW(ConjugateLikelihood):
    """
    Multivariate Gaussian with Normal–Inverse–Wishart prior (skeleton).

    Parameters
    ----------
    m0 : ArrayF, shape (d,)
    kappa0 : float (>0)
    nu0 : float (> d-1)
    Psi0 : ArrayF, shape (d,d), SPD
    """

    def __init__(self, m0: ArrayF, kappa0: float, nu0: float, Psi0: ArrayF) -> None:
        if kappa0 <= 0:
            raise ValueError("kappa0 must be > 0.")
        if Psi0.ndim != 2 or Psi0.shape[0] != Psi0.shape[1]:
            raise ValueError("Psi0 must be (d,d) SPD.")
        d = Psi0.shape[0]
        if nu0 <= d - 1:
            raise ValueError("nu0 must be > d - 1.")
        if m0.shape != (d,):
            raise ValueError("m0 must be shape (d,) matching Psi0.")

        self.m0 = np.asarray(m0, dtype=float)
        self.kappa0 = float(kappa0)
        self.nu0 = float(nu0)
        self.Psi0 = np.asarray(Psi0, dtype=float)
        self.d = d

        self.stats = GaussianNIWStats()

    def init_stats(self, R: int) -> None:
        d = self.d
        self.stats.m = np.repeat(self.m0[None, :], R, axis=0)
        self.stats.kappa = np.full(R, self.kappa0, dtype=float)
        self.stats.nu = np.full(R, self.nu0, dtype=float)
        self.stats.Psi = np.repeat(self.Psi0[None, :, :], R, axis=0)

    def predictive_prob(self, x_t, /) -> ArrayF:
        # TODO: return multivariate Student-t predictive densities per state
        raise NotImplementedError("GaussianNIW.predictive_prob is not yet implemented.")

    def prior_predictive_prob(self, x_t, /) -> float:
        raise NotImplementedError("GaussianNIW.prior_predictive_prob is not yet implemented.")

    def predictive_mean(self) -> ArrayF:
        # Posterior predictive mean equals m (the posterior mean of μ)
        if self.stats.m is None:
            raise RuntimeError("Call init_stats(R) before predictive_mean().")
        # Return the *component-wise* means; BOCPD may not use this directly for multivariate data.
        return self.stats.m  # type: ignore[return-value]

    def update_cp(self, x_t) -> None:
        # TODO: Update r=0 NIW params with one new x_t (rank-1 updates).
        raise NotImplementedError("GaussianNIW.update_cp is not yet implemented.")

    def update_growth(self, x_t) -> None:
        # TODO: Shift r -> r+1 and apply NIW posterior update with x_t to all grown states.
        raise NotImplementedError("GaussianNIW.update_growth is not yet implemented.")

