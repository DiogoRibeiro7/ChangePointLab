# sdhmm.py
# MIT License
# (c) 2025

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import numpy as np
from numpy.typing import NDArray

from ...core.datatypes import LatentStateResult
from ...core.segmentation import changepoints_from_labels, normalize_linear_changepoints
from .._base import BaseDetector

# Scientific traceability:
# - Manouchehri and Bouguila (2023), doi:10.3390/s23031390.
# - Registry entry: docs/science/method_registry.yml, method id "sd_hmm".

ArrayF = NDArray[np.floating]
ArrayI = NDArray[np.integer]


# ----------------------------- Utilities -----------------------------

def _as_float_array(x: np.ndarray, name: str) -> ArrayF:
    a = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values.")
    return a


def _normalize_rows_stable(mat: ArrayF, eps: float = 1e-12) -> ArrayF:
    """Normalize rows to sum to 1 (proportional vectors); clamps tiny totals."""
    s = mat.sum(axis=1, keepdims=True)
    s = np.where(s <= eps, 1.0, s)
    out = mat / s
    # avoid exact zeros (log-safe)
    return np.clip(out, eps, 1.0)


def _logsumexp(a: ArrayF, axis: Optional[int] = None) -> ArrayF:
    m = np.max(a, axis=axis, keepdims=True)
    z = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return z if axis is None else np.squeeze(z, axis=axis)


def _gammaln(x: ArrayF) -> ArrayF:
    # numpy has gammaln via scipy.special only; use math.lgamma elementwise
    vec = np.vectorize(math.lgamma)
    return vec(x)


def _digamma(x: ArrayF) -> ArrayF:
    """
    Fast digamma approximation (sufficient for optimization steps).
    Uses asymptotic expansion with recurrence to shift x > 6.
    """
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        # reflect small/invalid to a small positive to avoid NaNs
        x = np.clip(x, 1e-8, None)
    y = x.copy()
    # recurrence: psi(x) = psi(x+1) - 1/x
    k = np.zeros_like(y)
    while True:
        m = y < 6.0
        if not np.any(m):
            break
        y[m] += 1.0
        k[m] += 1.0
    # asymptotic expansion
    r = 1.0 / y
    r2 = r * r
    psi_asym = np.log(y) - 0.5 * r - r2 * (1.0/12.0 - r2 * (1.0/120.0 - r2 * (1.0/252.0)))
    # undo recurrence
    psi = psi_asym - k / x
    return psi


# ------------------------ Scaled-Dirichlet emission ------------------------

@dataclass
class SDParams:
    """Scaled-Dirichlet parameters for one state: alpha (D,), beta (D,) on the simplex."""
    alpha: ArrayF
    beta: ArrayF


def _sd_logpdf(x: ArrayF, alpha: ArrayF, beta: ArrayF) -> float:
    """
    Log pdf of the Scaled-Dirichlet at x (row vector, sum~1), parameters alpha,beta > 0.

    log p(x) = lgamma(sum alpha) - sum lgamma(alpha_d)
               + sum alpha_d * log beta_d
               + sum (alpha_d - 1) * log x_d
               - (sum alpha) * log( sum_d beta_d * x_d )
    """
    S = float(np.sum(alpha))
    lg = float(math.lgamma(S) - np.sum(_gammaln(alpha)))
    term_beta = float(np.dot(alpha, np.log(beta)))
    term_x = float(np.dot(alpha - 1.0, np.log(x)))
    denom = float(np.log(np.dot(beta, x)))
    return lg + term_beta + term_x - S * denom


def _sd_logpdf_batch(X: ArrayF, alpha: ArrayF, beta: ArrayF) -> ArrayF:
    """Vectorized log pdf for all rows of X."""
    S = float(np.sum(alpha))
    lg = float(math.lgamma(S) - np.sum(_gammaln(alpha)))
    term_beta = float(np.dot(alpha, np.log(beta)))
    term_x = (alpha - 1.0) @ np.log(X.T)  # shape (T,)
    denom = np.log(X @ beta)              # shape (T,)
    return lg + term_beta + term_x - S * denom


def _sd_grad_alpha_beta(
    X: ArrayF,  # (T, D), rows sum to 1
    w: ArrayF,  # (T,), responsibilities for this state, >=0
    alpha: ArrayF,  # (D,)
    beta: ArrayF,   # (D,)
    u0: float, v0: float,       # Gamma prior on alpha_d ~ Gamma(u0, v0)
    h0: ArrayF,                 # Dirichlet prior for beta ~ Dir(h0)
) -> Tuple[ArrayF, ArrayF]:
    """
    Gradient of (weighted) log posterior wrt alpha, beta (beta via softmax reparam handled outside).

    d/d alpha_d: sum_t w_t [ psi(sum alpha) - psi(alpha_d) + log beta_d + log x_td - log(beta·x_t) ] + (u0-1)/alpha_d - v0
    d/d beta_d : sum_t w_t [ alpha_d / beta_d - (sum alpha) * x_td / (beta·x_t) ] + (h0_d - 1)/beta_d  (before simplex constraint)
    """
    T, D = X.shape
    S = float(np.sum(alpha))
    psi_sum = float(_digamma(np.array([S]))[0])
    psi_alpha = _digamma(alpha)
    bx = X @ beta  # (T,)
    # guard against underflow
    bx = np.clip(bx, 1e-12, None)

    # alpha gradient
    log_beta = np.log(beta)
    g_alpha = np.empty(D, dtype=float)
    for d in range(D):
        term = np.sum(w * (psi_sum - psi_alpha[d] + log_beta[d] + np.log(X[:, d]) - np.log(bx)))
        g_alpha[d] = term + (u0 - 1.0) / max(alpha[d], 1e-12) - v0

    # beta gradient (unconstrained form)
    g_beta = np.zeros(D, dtype=float)
    for d in range(D):
        term1 = np.sum(w) * alpha[d] / max(beta[d], 1e-12)
        term2 = np.sum(w * (S * X[:, d] / bx))
        prior = (h0[d] - 1.0) / max(beta[d], 1e-12)
        g_beta[d] = term1 - term2 + prior

    return g_alpha, g_beta


# ----------------------------- HMM Model -----------------------------

@dataclass
class SDHMMConfig:
    K: int                   # number of hidden states
    max_iter: int = 100
    tol: float = 1e-4
    min_iter: int = 10
    seed: Optional[int] = 123
    # priors (mild, to stabilize)
    trans_dirichlet: float = 1.0
    init_dirichlet: float = 1.0
    alpha_gamma_u0: float = 2.0
    alpha_gamma_v0: float = 1.0
    beta_dirichlet_h0: float = 1.0
    # emission optim steps
    em_steps: int = 5
    lr_alpha: float = 0.05
    lr_beta: float = 0.05


@dataclass
class SDHMMResult:
    pi: ArrayF                 # (K,)
    A: ArrayF                  # (K,K)
    params: Tuple[SDParams, ...]
    loglik: float
    gamma: ArrayF              # (T,K) posterior state probs
    xi: ArrayF                 # (T-1,K,K) expected transitions


class _SDHMM:
    """
    Scaled-Dirichlet HMM (unsupervised) with MAP/variational learning.

    - Observations X are proportional feature vectors (each row sums to ~1).
    - Emission per state j: Scaled-Dirichlet(alpha_j, beta_j), alpha_j>0, beta_j>0 on simplex.
    - Learning:
        * E-step: forward–backward with emission log-likelihood at current params.
        * M-step: Dirichlet posteriors for pi, A; MAP gradient steps for alpha/beta.

    Reference: SD-HMM idea and scaled-Dirichlet density from Manouchehri and
    Bouguila (2023).
    """

    def __init__(self, cfg: SDHMMConfig) -> None:
        if cfg.K < 1:
            raise ValueError("K must be >= 1.")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # learned state
        self.pi: ArrayF | None = None
        self.A: ArrayF | None = None
        self.params: tuple[SDParams, ...] | None = None

    # ---------- initialization ----------

    def _init_params(self, X: ArrayF) -> None:
        T, D = X.shape
        K = self.cfg.K

        # init pi, A as Dirichlet draws
        self.pi = np.full(K, 1.0 / K, dtype=float)
        A = self.rng.dirichlet(np.full(K, 1.0), size=K)
        self.A = A

        # kmeans-ish init by random partitions
        z = self.rng.integers(0, K, size=T)
        params: list[SDParams] = []
        for j in range(K):
            Xj = X[z == j]
            if Xj.size == 0:
                Xj = X
            # start alpha near 2.0 (mild concentration), beta ~ average composition
            alpha0 = np.full(D, 2.0, dtype=float)
            beta0 = np.clip(Xj.mean(axis=0), 1e-3, None)
            beta0 = beta0 / beta0.sum()
            params.append(SDParams(alpha=alpha0, beta=beta0))
        self.params = tuple(params)

    # ---------- core emissions ----------

    def _log_emission_matrix(self, X: ArrayF) -> ArrayF:
        """Return log B[t, j] = log p(x_t | state j)."""
        assert self.params is not None
        T = X.shape[0]
        K = self.cfg.K
        out = np.empty((T, K), dtype=float)
        for j, p in enumerate(self.params):
            out[:, j] = _sd_logpdf_batch(X, p.alpha, p.beta)
        return out

    # ---------- forward-backward (scaled) ----------

    def _forward_backward(self, logB: ArrayF) -> Tuple[ArrayF, ArrayF, ArrayF, float]:
        """
        Returns (gamma, xi, log_alpha_T, loglik)
        gamma: (T,K), xi: (T-1,K,K)
        """
        assert self.pi is not None and self.A is not None
        pi, A = self.pi, self.A
        T, K = logB.shape
        logA = np.log(A + 1e-300)
        logpi = np.log(pi + 1e-300)

        # forward
        log_alpha = np.empty((T, K), dtype=float)
        log_alpha[0] = logpi + logB[0]
        for t in range(1, T):
            prev = log_alpha[t - 1][:, None] + logA  # (K,K)
            log_alpha[t] = _logsumexp(prev, axis=0) + logB[t]

        # backward
        log_beta = np.zeros((T, K), dtype=float)
        for t in range(T - 2, -1, -1):
            tmp = logA + logB[t + 1][None, :] + log_beta[t + 1][None, :]
            log_beta[t] = _logsumexp(tmp, axis=1)

        # posteriors
        log_gamma = log_alpha + log_beta
        ll = float(_logsumexp(log_alpha[-1], axis=0))
        log_gamma -= ll
        gamma = np.exp(log_gamma)

        # xi
        xi = np.empty((T - 1, K, K), dtype=float)
        for t in range(T - 1):
            m = log_alpha[t][:, None] + logA + logB[t + 1][None, :] + log_beta[t + 1][None, :]
            m -= _logsumexp(m, axis=None)  # normalize over KxK
            xi[t] = np.exp(m)

        return gamma, xi, log_alpha[-1], ll

    # ---------- M-step: transitions ----------

    def _mstep_transitions(self, gamma: ArrayF, xi: ArrayF) -> None:
        assert self.A is not None and self.pi is not None
        # priors
        a0 = float(self.cfg.trans_dirichlet)
        p0 = float(self.cfg.init_dirichlet)

        # initial
        pi_post = p0 + gamma[0]
        self.pi = pi_post / pi_post.sum()

        # transitions
        A_counts = xi.sum(axis=0)  # (K,K)
        A_post = A_counts + a0
        self.A = (A_post.T / A_post.sum(axis=1)).T  # row-normalize

    # ---------- M-step: emissions (per state) ----------

    def _mstep_emissions(self, X: ArrayF, gamma: ArrayF) -> None:
        assert self.params is not None
        T, D = X.shape
        u0, v0 = float(self.cfg.alpha_gamma_u0), float(self.cfg.alpha_gamma_v0)
        h0 = np.full(D, float(self.cfg.beta_dirichlet_h0), dtype=float)

        new_params: list[SDParams] = []
        for j, p in enumerate(self.params):
            w = gamma[:, j]
            w = w / (w.sum() + 1e-12)

            alpha = p.alpha.copy()
            beta = p.beta.copy()

            for _ in range(self.cfg.em_steps):
                # gradients
                g_alpha, g_beta_uncon = _sd_grad_alpha_beta(X, w, alpha, beta, u0, v0, h0)

                # gradient steps with positivity projection for alpha
                alpha = np.clip(alpha + self.cfg.lr_alpha * g_alpha, 1e-5, 1e9)

                # update beta in softmax parameterization for simplex constraint:
                # beta = softmax(wb), do gradient ascent on wb using chain rule.
                # For simplicity, transform to logits, take step in unconstrained space using g_beta on beta
                # with mirror descent-like update: beta <- normalize( beta * exp(lr * g_beta_uncon) )
                step = np.exp(self.cfg.lr_beta * g_beta_uncon)
                beta = beta * step
                beta = np.clip(beta, 1e-12, None)
                beta /= beta.sum()

            new_params.append(SDParams(alpha=alpha, beta=beta))

        self.params = tuple(new_params)

    # ---------- Fit ----------

    def fit(self, X: ArrayF) -> SDHMMResult:
        """
        Fit the SD-HMM with unsupervised learning.

        Parameters
        ----------
        X : array of shape (T, D)
            Each row is a non-negative feature vector; will be normalized to sum to 1 per row.

        Returns
        -------
        SDHMMResult
        """
        X = _as_float_array(np.asarray(X), "X")
        if X.ndim != 2:
            raise ValueError("X must be 2-D (T, D).")
        if np.any(X < 0):
            raise ValueError("X must be non-negative.")
        X = _normalize_rows_stable(X)

        # init
        self._init_params(X)

        prev_ll = -np.inf
        ll_trace = []

        for it in range(self.cfg.max_iter):
            # E-step
            logB = self._log_emission_matrix(X)
            gamma, xi, _, ll = self._forward_backward(logB)
            ll_trace.append(ll)

            # M-step
            self._mstep_transitions(gamma, xi)
            self._mstep_emissions(X, gamma)

            # stopping
            if it + 1 >= self.cfg.min_iter:
                if abs(ll - prev_ll) < self.cfg.tol * (1.0 + abs(prev_ll)):
                    break
            prev_ll = ll

        assert self.pi is not None and self.A is not None and self.params is not None
        return SDHMMResult(
            pi=self.pi.copy(),
            A=self.A.copy(),
            params=self.params,
            loglik=float(ll_trace[-1]),
            gamma=gamma,
            xi=xi,
        )

    # ---------- Decoding ----------

    def viterbi(self, X: ArrayF) -> ArrayI:
        """Most likely state sequence."""
        X = _normalize_rows_stable(_as_float_array(X, "X"))
        assert self.pi is not None and self.A is not None and self.params is not None
        T = X.shape[0]
        K = self.cfg.K
        logB = self._log_emission_matrix(X)
        logA = np.log(self.A + 1e-300)
        logpi = np.log(self.pi + 1e-300)

        delta = np.empty((T, K), dtype=float)
        psi = np.empty((T, K), dtype=int)

        delta[0] = logpi + logB[0]
        psi[0] = -1
        for t in range(1, T):
            M = delta[t - 1][:, None] + logA
            psi[t] = np.argmax(M, axis=0)
            delta[t] = np.max(M, axis=0) + logB[t]

        z = np.empty(T, dtype=int)
        z[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            z[t] = psi[t + 1, z[t + 1]]
        return z


class SDHMM(_SDHMM, BaseDetector):
    """SD-HMM wrapper exposing fit/predict methods."""

    def fit(self, X: ArrayF) -> SDHMM:  # type: ignore[override]
        self._validate_input(X)
        self.result_ = super().fit(X)
        self._X = X
        return self

    def predict(self, X: ArrayF | None = None) -> LatentStateResult:
        if X is not None:
            return self.fit(X).predict()
        if not hasattr(self, "_X"):
            raise RuntimeError("Call fit before predict.")
        states = super().viterbi(self._X)
        cps = normalize_linear_changepoints(
            changepoints_from_labels(states),
            n=states.size,
        )
        provenance = {
            "seed": self.cfg.seed,
            "rng": "numpy.random.Generator",
            "K": self.cfg.K,
            "max_iter": self.cfg.max_iter,
            "min_iter": self.cfg.min_iter,
            "tol": self.cfg.tol,
        }
        meta = {"states": states, "result": self.result_, "provenance": provenance}
        return LatentStateResult(
            indices=cps,
            method_name="sdhmm",
            states=states,
            metadata=meta,
            provenance=provenance,
        )


__all__ = ["SDHMM", "SDHMMConfig", "SDHMMResult", "SDParams"]
