# sdhmm_mix_vi.py
# MIT License
# (c) 2025

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import math
import numpy as np
from numpy.typing import NDArray

from ...core.datatypes import ChangePointResult
from .._base import BaseDetector

# Scientific traceability:
# - Manouchehri and Bouguila (2023), doi:10.3390/s23031390.
# - Registry entry: docs/science/method_registry.yml, method id "sd_hmm_mix_vi".

ArrayF = NDArray[np.floating]
ArrayI = NDArray[np.integer]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _as_float_array(x: np.ndarray, name: str) -> ArrayF:
    a = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values.")
    return a


def _normalize_rows_stable(mat: ArrayF, eps: float = 1e-12) -> ArrayF:
    s = mat.sum(axis=1, keepdims=True)
    s = np.where(s <= eps, 1.0, s)
    out = mat / s
    return np.clip(out, eps, 1.0)


def _logsumexp(a: ArrayF, axis: Optional[int] = None) -> ArrayF:
    m = np.max(a, axis=axis, keepdims=True)
    z = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return z if axis is None else np.squeeze(z, axis=axis)


def _gammaln(x: ArrayF) -> ArrayF:
    vec = np.vectorize(math.lgamma)
    return vec(x)


def _digamma(x: ArrayF) -> ArrayF:
    """
    Fast digamma approximation (stable for optimization).
    Uses recurrence to shift x>6 and an asymptotic series.
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 1e-12, None)
    y = x.copy()
    k = np.zeros_like(y)
    while True:
        m = y < 6.0
        if not np.any(m):
            break
        y[m] += 1.0
        k[m] += 1.0
    r = 1.0 / y
    r2 = r * r
    psi_asym = np.log(y) - 0.5 * r - r2 * (1.0/12.0 - r2 * (1.0/120.0 - r2 * (1.0/252.0)))
    psi = psi_asym - k / x
    return psi


# ---------------------------------------------------------------------
# Scaled-Dirichlet emission
# ---------------------------------------------------------------------

@dataclass
class SDMixParams:
    """
    Parameters of a single Scaled-Dirichlet component:
      alpha: (D,) strictly positive
      beta : (D,) on the simplex
    """
    alpha: ArrayF
    beta: ArrayF


def _sd_logpdf_batch(X: ArrayF, alpha: ArrayF, beta: ArrayF) -> ArrayF:
    """
    Vectorized log pdf for all rows of X under Scaled-Dirichlet(alpha,beta).
    log p(x) = lgamma(sum alpha) - sum lgamma(alpha_d)
               + sum alpha_d log beta_d + sum (alpha_d-1) log x_d
               - (sum alpha) log(beta · x)
    """
    S = float(np.sum(alpha))
    lg = float(math.lgamma(S) - np.sum(_gammaln(alpha)))
    term_beta = float(np.dot(alpha, np.log(beta)))
    term_x = (alpha - 1.0) @ np.log(X.T)         # (T,)
    denom = np.log(np.clip(X @ beta, 1e-12, None))  # (T,)
    return lg + term_beta + term_x - S * denom


def _sd_grad_alpha_beta(
    X: ArrayF,             # (T, D) normalized rows (sum=1)
    w: ArrayF,             # (T,) nonnegative responsibilities for this component
    alpha: ArrayF,         # (D,)
    beta: ArrayF,          # (D,)
    u0: float, v0: float,  # Gamma(u0,v0) prior on each alpha_d
    h0: ArrayF,            # Dirichlet(h0) prior for beta
) -> Tuple[ArrayF, ArrayF]:
    """
    Gradient of expected log joint wrt alpha,beta given weights w_t.
    See derivation used in the simpler single-component model.

    d/d alpha_d:
        sum_t w_t [ psi(sum alpha) - psi(alpha_d) + log beta_d + log x_td - log(beta·x_t) ]
        + (u0-1)/alpha_d - v0

    d/d beta_d (unconstrained):
        sum_t w_t [ alpha_d / beta_d - (sum alpha) * x_td / (beta·x_t) ] + (h0_d - 1)/beta_d
    """
    T, D = X.shape
    S = float(np.sum(alpha))
    psi_sum = float(_digamma(np.array([S]))[0])
    psi_alpha = _digamma(alpha)
    bx = np.clip(X @ beta, 1e-12, None)  # (T,)

    g_alpha = np.empty(D, dtype=float)
    log_beta = np.log(beta)
    for d in range(D):
        term = np.sum(w * (psi_sum - psi_alpha[d] + log_beta[d] + np.log(X[:, d]) - np.log(bx)))
        g_alpha[d] = term + (u0 - 1.0) / max(alpha[d], 1e-12) - v0

    g_beta = np.empty(D, dtype=float)
    for d in range(D):
        term1 = np.sum(w) * alpha[d] / max(beta[d], 1e-12)
        term2 = np.sum(w * (S * X[:, d] / bx))
        prior = (h0[d] - 1.0) / max(beta[d], 1e-12)
        g_beta[d] = term1 - term2 + prior

    return g_alpha, g_beta


# ---------------------------------------------------------------------
# Model config / result
# ---------------------------------------------------------------------

@dataclass
class SDHMMMixVIConfig:
    K: int                           # states
    M: int                           # components per state (uniform for simplicity)
    max_iter: int = 100
    min_iter: int = 10
    tol: float = 1e-4
    seed: Optional[int] = 123

    # Priors
    trans_dirichlet: float = 1.0     # A row prior (symmetric)
    init_dirichlet: float = 1.0      # pi prior (symmetric)
    mix_dirichlet: float = 1.0       # w_j prior (symmetric)
    alpha_gamma_u0: float = 2.0      # Gamma prior for alpha_d
    alpha_gamma_v0: float = 1.0
    beta_dirichlet_h0: float = 1.0   # Dirichlet prior for beta

    # Emission optimization (per VI iteration)
    em_steps: int = 5
    lr_alpha: float = 0.05
    lr_beta: float = 0.05


@dataclass
class SDHMMMixVIResult:
    pi: ArrayF                 # (K,)
    A: ArrayF                  # (K,K)
    eta: ArrayF                # (K,M) Dirichlet params for mixture weights
    params: Tuple[Tuple[SDMixParams, ...], ...]  # (K, M) components
    loglik: float
    gamma: ArrayF              # (T,K) posterior state probs
    r: ArrayF                  # (T,K,M) mixture responsibilities (conditional on state j)
    xi: ArrayF                 # (T-1,K,K)


# ---------------------------------------------------------------------
# VI SD-HMM with per-state mixtures
# ---------------------------------------------------------------------

class _SDHMMMixVI:
    """
    Scaled-Dirichlet HMM with per-state mixtures and VI.
    - States j=1..K, each with M components.
    - q(S_{1:T}) via forward–backward using effective state emissions.
    - q(C_t | S_t=j) via soft assignments with E[log w_jm].
    - q(w_j) = Dir(eta_j) with eta_j updated from expected counts.
    - Emission params per (j,m) updated by gradient ascent on ELBO using weights w_{tjm} = gamma_{tj} * r_{tjm}.
    """

    def __init__(self, cfg: SDHMMMixVIConfig):
        if cfg.K < 1 or cfg.M < 1:
            raise ValueError("K and M must be >= 1.")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # learned
        self.pi: ArrayF | None = None         # (K,)
        self.A: ArrayF | None = None          # (K,K)
        self.eta: ArrayF | None = None        # (K,M)
        self.params: Tuple[Tuple[SDMixParams, ...], ...] | None = None  # (K,M)

    # ----------------- init -----------------

    def _init_params(self, X: ArrayF) -> None:
        T, D = X.shape
        K, M = self.cfg.K, self.cfg.M

        # Initialize pi, A
        self.pi = np.full(K, 1.0 / K, dtype=float)
        self.A = self.rng.dirichlet(np.full(K, 1.0), size=K)

        # Mixture Dirichlet params
        self.eta = np.full((K, M), float(self.cfg.mix_dirichlet), dtype=float)

        # Rough partition for seeding components
        z_state = self.rng.integers(0, K, size=T)
        z_comp = self.rng.integers(0, M, size=T)

        params: List[Tuple[SDMixParams, ...]] = []
        for j in range(K):
            comps: List[SDMixParams] = []
            Xj = X[z_state == j]
            if Xj.size == 0:
                Xj = X
            for m in range(M):
                Xm = Xj[z_comp[z_state == j] == m] if Xj.size > 0 else X
                if Xm.size == 0:
                    Xm = Xj
                # start alphas around 2.0, betas near mean composition
                alpha0 = np.full(D, 2.0, dtype=float)
                beta0 = np.clip(Xm.mean(axis=0), 1e-3, None)
                beta0 = beta0 / beta0.sum()
                comps.append(SDMixParams(alpha=alpha0, beta=beta0))
            params.append(tuple(comps))
        self.params = tuple(params)

    # ----------------- expected logs -----------------

    @staticmethod
    def _elog_dirichlet_row(alpha: ArrayF) -> ArrayF:
        """E[log θ_i] for Dirichlet(alpha) row vector."""
        s = float(np.sum(alpha))
        return _digamma(alpha) - _digamma(np.array([s]))[0]

    # ----------------- emissions -----------------

    def _log_emission_tensor(self, X: ArrayF) -> ArrayF:
        """Return log E[t,j,m] = log p(x_t | component (j,m)); shape (T,K,M)."""
        assert self.params is not None
        T = X.shape[0]
        K, M = self.cfg.K, self.cfg.M
        E = np.empty((T, K, M), dtype=float)
        for j in range(K):
            for m in range(M):
                p = self.params[j][m]
                E[:, j, m] = _sd_logpdf_batch(X, p.alpha, p.beta)
        return E

    # ----------------- forward-backward with mixtures collapsed -----------------

    def _forward_backward(self, logB_state: ArrayF) -> Tuple[ArrayF, ArrayF, float]:
        """
        Run FB on state chain given per-state log emissions.
        Returns (gamma, xi, loglik).
        """
        assert self.pi is not None and self.A is not None
        T, K = logB_state.shape
        logA = np.log(self.A + 1e-300)
        logpi = np.log(self.pi + 1e-300)

        # forward
        log_alpha = np.empty((T, K), dtype=float)
        log_alpha[0] = logpi + logB_state[0]
        for t in range(1, T):
            prev = log_alpha[t - 1][:, None] + logA
            log_alpha[t] = _logsumexp(prev, axis=0) + logB_state[t]

        # backward
        log_beta = np.zeros((T, K), dtype=float)
        for t in range(T - 2, -1, -1):
            tmp = logA + logB_state[t + 1][None, :] + log_beta[t + 1][None, :]
            log_beta[t] = _logsumexp(tmp, axis=1)

        # posteriors
        ll = float(_logsumexp(log_alpha[-1], axis=0))
        log_gamma = log_alpha + log_beta - ll
        gamma = np.exp(log_gamma)

        # xi
        xi = np.empty((T - 1, K, K), dtype=float)
        for t in range(T - 1):
            m = log_alpha[t][:, None] + logA + logB_state[t + 1][None, :] + log_beta[t + 1][None, :]
            m -= _logsumexp(m, axis=None)
            xi[t] = np.exp(m)

        return gamma, xi, ll

    # ----------------- VI loop -----------------

    def fit(self, X: ArrayF) -> SDHMMMixVIResult:
        """
        Fit per-state mixture SD-HMM by variational inference.

        Parameters
        ----------
        X : array (T, D), nonnegative; rows will be normalized to sum to 1.
        """
        X = _as_float_array(X, "X")
        if X.ndim != 2:
            raise ValueError("X must be 2-D (T, D).")
        if np.any(X < 0):
            raise ValueError("X must be non-negative.")
        X = _normalize_rows_stable(X)

        # init
        self._init_params(X)
        assert self.params is not None and self.eta is not None

        T, D = X.shape
        K, M = self.cfg.K, self.cfg.M

        a0 = float(self.cfg.trans_dirichlet)
        p0 = float(self.cfg.init_dirichlet)
        eta0 = float(self.cfg.mix_dirichlet)
        u0, v0 = float(self.cfg.alpha_gamma_u0), float(self.cfg.alpha_gamma_v0)
        h0 = np.full(D, float(self.cfg.beta_dirichlet_h0), dtype=float)

        prev_ll = -np.inf
        for it in range(self.cfg.max_iter):
            # ----- E-step -----
            # (1) expected log weights per state
            elog_w = np.empty((K, M), dtype=float)
            for j in range(K):
                elog_w[j] = self._elog_dirichlet_row(self.eta[j])

            # (2) component-wise log emissions
            logE = self._log_emission_tensor(X)  # (T,K,M)

            # (3) effective per-state log emission: logsumexp over mixtures with elog_w
            logB_state = np.empty((T, K), dtype=float)
            for j in range(K):
                logB_state[:, j] = _logsumexp(logE[:, j, :] + elog_w[j][None, :], axis=1)

            # (4) forward-backward on states
            gamma, xi, ll = self._forward_backward(logB_state)

            # (5) mixture responsibilities r_{tjm} (conditional within each state j)
            # r_{tjm} ∝ exp(elog_w[jm] + logE[t,j,m])
            r = np.empty((T, K, M), dtype=float)
            for j in range(K):
                tmp = logE[:, j, :] + elog_w[j][None, :]
                tmp -= _logsumexp(tmp, axis=1)[:, None]
                r[:, j, :] = np.exp(tmp)

            # expected component counts per state
            N_jm = np.sum(gamma[:, :, None] * r, axis=0)  # (K,M)

            # ----- M-step (variational) -----
            # (1) initial and transition posteriors (Dirichlet)
            pi_post = p0 + gamma[0]
            self.pi = pi_post / pi_post.sum()

            A_counts = xi.sum(axis=0)                   # (K,K)
            A_post = A_counts + a0
            self.A = (A_post.T / A_post.sum(axis=1)).T  # row-normalize

            # (2) mixture weights posteriors (Dirichlet)
            self.eta = eta0 + N_jm

            # (3) emissions: for each (j,m), ascend ELBO with weights w_{tjm} = gamma_tj * r_tjm
            for j in range(K):
                for m in range(M):
                    comp = self.params[j][m]
                    wt = gamma[:, j] * r[:, j, m]  # (T,)
                    # normalize weights for numerical stability of steps (objective is linear in weights)
                    scale = wt.sum() + 1e-12
                    wnorm = wt / scale

                    alpha = comp.alpha.copy()
                    beta = comp.beta.copy()

                    for _ in range(self.cfg.em_steps):
                        g_alpha, g_beta_unc = _sd_grad_alpha_beta(
                            X, wnorm, alpha, beta, u0, v0, h0
                        )
                        # simple projected ascent
                        alpha = np.clip(alpha + self.cfg.lr_alpha * g_alpha, 1e-5, 1e9)

                        # mirror-descent style update for simplex beta
                        step = np.exp(self.cfg.lr_beta * g_beta_unc)
                        beta = np.clip(beta * step, 1e-12, None)
                        beta /= beta.sum()

                    # commit
                    self.params[j] = tuple(
                        SDMixParams(alpha=alpha, beta=beta) if mm == m else self.params[j][mm]
                        for mm in range(M)
                    )

            # ----- stopping -----
            if it + 1 >= self.cfg.min_iter:
                if abs(ll - prev_ll) < self.cfg.tol * (1.0 + abs(prev_ll)):
                    break
            prev_ll = ll

        assert self.pi is not None and self.A is not None and self.params is not None and self.eta is not None
        return SDHMMMixVIResult(
            pi=self.pi.copy(),
            A=self.A.copy(),
            eta=self.eta.copy(),
            params=self.params,
            loglik=float(prev_ll),
            gamma=gamma,
            r=r,
            xi=xi,
        )

    # ----------------- decoders -----------------

    def viterbi_states(self, X: ArrayF) -> ArrayI:
        """
        Viterbi decoding for states (mixtures marginalized).
        Returns z (T,) with most likely state path.
        """
        X = _normalize_rows_stable(_as_float_array(X, "X"))
        assert self.pi is not None and self.A is not None and self.params is not None and self.eta is not None
        T = X.shape[0]
        K, M = self.cfg.K, self.cfg.M

        # build per-state effective log emissions as in fit()
        elog_w = np.empty((K, M), dtype=float)
        for j in range(K):
            elog_w[j] = self._elog_dirichlet_row(self.eta[j])
        logE = self._log_emission_tensor(X)
        logB = np.empty((T, K), dtype=float)
        for j in range(K):
            logB[:, j] = _logsumexp(logE[:, j, :] + elog_w[j][None, :], axis=1)

        logA = np.log(self.A + 1e-300)
        logpi = np.log(self.pi + 1e-300)

        delta = np.empty((T, K), dtype=float)
        psi = np.empty((T, K), dtype=int)

        delta[0] = logpi + logB[0]
        psi[0] = -1
        for t in range(1, T):
            Mtx = delta[t - 1][:, None] + logA
            psi[t] = np.argmax(Mtx, axis=0)
            delta[t] = np.max(Mtx, axis=0) + logB[t]

        z = np.empty(T, dtype=int)
        z[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            z[t] = psi[t + 1, z[t + 1]]
        return z

    def most_likely_components(self, X: ArrayF) -> ArrayI:
        """
        Per time step, return the most likely mixture component (m*) within the most likely state.
        Useful for quick inspection; prefer gamma/r for soft usage.
        """
        z = self.viterbi_states(X)
        Xn = _normalize_rows_stable(_as_float_array(X, "X"))
        assert self.params is not None and self.eta is not None
        T = Xn.shape[0]
        M = self.cfg.M
        m_hat = np.empty(T, dtype=int)

        # E[log w] for chosen state each t
        elog_w_all = [self._elog_dirichlet_row(self.eta[j]) for j in range(self.cfg.K)]
        for t in range(T):
            j = int(z[t])
            logE = np.array([_sd_logpdf_batch(Xn[t:t+1], self.params[j][m].alpha, self.params[j][m].beta)[0]
                             for m in range(M)])
            scores = elog_w_all[j] + logE
            m_hat[t] = int(np.argmax(scores))
        return m_hat


class SDHMMMixVI(_SDHMMMixVI, BaseDetector):
    """SD-HMM mixture VI wrapper with sklearn-like API."""

    def fit(self, X: ArrayF) -> SDHMMMixVI:
        self._validate_input(X)
        self.result_ = super().fit(X)
        self._X = X
        return self

    def predict(self, X: ArrayF | None = None) -> ChangePointResult:
        if X is not None:
            return self.fit(X).predict()
        if not hasattr(self, "_X"):
            raise RuntimeError("Call fit before predict.")
        states = super().viterbi_states(self._X)
        cps = np.flatnonzero(np.diff(states)) + 1
        meta = {"states": states, "result": self.result_}
        return ChangePointResult(indices=cps, metadata=meta)


__all__ = [
    "SDHMMMixVI",
    "SDHMMMixVIConfig",
    "SDHMMMixVIResult",
    "SDMixParams",
]
