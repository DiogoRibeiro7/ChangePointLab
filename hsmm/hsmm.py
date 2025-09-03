"""
Hidden semi-Markov models with explicit duration.

References
----------
.. [1] S.-Z. Yu (2010). "Hidden semi-Markov models." *Artificial Intelligence* 174(2):215-243.
.. [2] M. J. Johnson et al. (2015). "Composing graphical models with neural networks for structured representations and fast inference." *Advances in Neural Information Processing Systems*.
"""

# hsmm.py
# MIT License
# (c) 2025

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.floating]
ArrayI = NDArray[np.integer]
LOGZERO = float("-inf")


def _as_scalar(x: ArrayF | float) -> float:
    """Return a Python float from a NumPy scalar or array."""
    if isinstance(x, np.ndarray):
        return x.item()
    return float(x)


# ----------------------------- small math helpers -----------------------------

def logsumexp(a: ArrayF, axis: Optional[int] = None) -> ArrayF:
    """Stable logsumexp."""
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return out if axis is None else np.squeeze(out, axis=axis)


def safe_log(x: ArrayF | float, eps: float = 1e-300) -> ArrayF | float:
    return np.log(np.clip(x, eps, None))


# ----------------------------- duration models -----------------------------

@dataclass
class PoissonDur:
    """Zero-truncated, upper-truncated Poisson (support d∈[1, Dmax])."""
    lam: ArrayF  # (K,)

@dataclass
class NegBinDur:
    """
    Negative Binomial with support d∈[1, Dmax]; parameters:
      r > 0 (shape/dispersion), p in (0,1) (success prob for 'count' parameterization).
    We update (r,p) by robust MoM; likelihood uses proper truncation when evaluating.
    """
    r: ArrayF    # (K,)
    p: ArrayF    # (K,)

DurationModel = Tuple[str, object]  # ("poisson", PoissonDur) or ("negbin", NegBinDur)


def _poisson_logpmf_trunc(d: ArrayI, lam: float, Dmax: int) -> ArrayF:
    """log P(d | lam) for d∈[1,Dmax], normalized over [1..Dmax]."""
    d = np.asarray(d, dtype=int)
    base = -lam + d * math.log(lam) - _log_factorial(d)
    # log Z(λ) = log sum_{1..Dmax} e^{-λ} λ^d / d!
    # compute in log-space
    ds = np.arange(1, Dmax + 1, dtype=int)
    base_all = -lam + ds * math.log(lam) - _log_factorial(ds)
    logZ = logsumexp(base_all)
    return base - logZ


def _poisson_dlogZ_terms(lam: float, Dmax: int) -> Tuple[float, float]:
    """
    Compute Z'(λ)/Z and Z''(λ)/Z needed for Newton updates.
    For f_d = e^{-λ} λ^d / d!, we have:
      d/dλ log f_d = d/λ - 1
      Z' = sum f_d * (d/λ - 1)
      Z'' = sum f_d * [ (d/λ - 1)^2 - d/λ^2 ]
    Return (Z'/Z, Z''/Z).
    """
    ds = np.arange(1, Dmax + 1, dtype=float)
    logf = -lam + ds * math.log(lam) - _log_factorial(ds)
    w = np.exp(logf - logsumexp(logf))  # normalized weights
    t1 = ds / lam - 1.0
    z1 = _as_scalar(np.dot(w, t1))
    z2 = _as_scalar(np.dot(w, (t1 * t1) - (ds / (lam * lam))))
    return z1, z2


def _log_factorial(n: ArrayI) -> ArrayF:
    """log(n!) via lgamma(n+1)."""
    return np.vectorize(math.lgamma)(n + 1.0)


def _negbin_logpmf_untrunc(d: ArrayI, r: float, p: float) -> ArrayF:
    """
    NB pmf on counts k=d with k≥0:  C(k+r-1,k) (1-p)^r p^k.
    We'll evaluate for d≥1 and later renormalize over [1..Dmax].
    """
    d = np.asarray(d, dtype=float)
    return (np.vectorize(math.lgamma)(d + r)
            - math.lgamma(r) - _log_factorial(d)
            + d * math.log(p) + r * math.log(1.0 - p))


def _negbin_logpmf_trunc(d: ArrayI, r: float, p: float, Dmax: int) -> ArrayF:
    """log P(d | r,p) truncated to d∈[1..Dmax]."""
    ds = np.arange(1, Dmax + 1, dtype=int)
    log_all = _negbin_logpmf_untrunc(ds, r, p)
    logZ = logsumexp(log_all)
    return _negbin_logpmf_untrunc(np.asarray(d, dtype=int), r, p) - logZ


# ----------------------------- HSMM configuration -----------------------------

@dataclass
class HSMMConfig:
    K: int
    Dmax: int                        # maximum duration considered (per state)
    min_duration: int = 1            # minimum duration (enforced in recursions)
    learn_durations: bool = True
    forbid_self_transitions: bool = True  # typical HSMM: dwell times handle self-stays
    max_em_iters: int = 50
    tol: float = 1e-5
    seed: Optional[int] = 123


@dataclass
class HSMMSufficient:
    """
    Sufficient expectations from E-step for duration updates and transitions.
    """
    # duration posteriors: eta[t, j, d] mass that a segment in state j of length d ends at t
    eta: NDArray[np.floating]      # shape (T, K, Dmax_masked) where d starts at 1
    # expected number of segments per state
    seg_count: ArrayF              # (K,)
    # expected total duration per state
    seg_total_dur: ArrayF          # (K,)
    # expected squared duration per state (for NB MoM)
    seg_total_d2: ArrayF           # (K,)
    # expected initial segments per state
    pi_counts: ArrayF              # (K,)
    # expected transition counts between states (i→j)
    xi_counts: ArrayF              # (K, K)
    # occupancy per time/state (optional; derived from eta)
    gamma: ArrayF                  # (T, K)
    # sequence log-likelihood
    loglik: float


@dataclass
class HSMMParams:
    """
    Model parameters (transitions + durations). Emissions provided externally via log-likelihoods.
    """
    pi: ArrayF       # (K,)
    A: ArrayF        # (K,K)
    duration: DurationModel


# ----------------------------- HSMM core -----------------------------

class HSMM:
    """
    Explicit-duration HSMM in log space following Yu (2010) and Johnson et al. (2015).

    You pass a (T, K) matrix of per-time per-state log-likelihoods, L[t, j] = log p(y_t | state=j).
    Emissions are not learned here; we focus on durations + transitions.

    References
    ----------
    .. [1] S.-Z. Yu (2010). "Hidden semi-Markov models." *Artificial Intelligence* 174(2):215-243.
    .. [2] M. J. Johnson et al. (2015). "Composing graphical models with neural networks for structured representations and fast inference." *Advances in Neural Information Processing Systems*.
    """

    def __init__(self, cfg: HSMMConfig, params: HSMMParams) -> None:
        self.cfg = cfg
        self.params = params
        self.rng = np.random.default_rng(cfg.seed)
        self._validate_and_prepare()
        self._dur_cache: Dict[int, ArrayF] = {}

    # -------------------- public API --------------------

    def fit(self, loglik_tk: ArrayF) -> Tuple[HSMMParams, List[float]]:
        """
        EM on durations + transitions. Emissions fixed via loglik_tk.

        Returns (updated_params, loglik_trace).
        """
        loglik_tk = self._check_loglik(loglik_tk)
        ll_trace: List[float] = []

        for it in range(self.cfg.max_em_iters):
            suff = self._e_step(loglik_tk)
            ll_trace.append(suff.loglik)

            if self.cfg.learn_durations:
                self._m_step_durations(suff)

            self._m_step_transitions(suff)

            if it > 0 and abs(ll_trace[-1] - ll_trace[-2]) < self.cfg.tol * (1 + abs(ll_trace[-2])):
                break

        return self.params, ll_trace

    def decode_viterbi(self, loglik_tk: ArrayF) -> Tuple[ArrayI, ArrayI]:
        """
        HSMM Viterbi (max-product). Returns (states z_t, durations d_t of segments aligned to ends).
        z_t is length T; durations per segment can be reconstructed from backpointers.
        """
        loglik_tk = self._check_loglik(loglik_tk)
        return self._viterbi(loglik_tk)

    # -------------------- internals --------------------

    def _validate_and_prepare(self) -> None:
        K = self.cfg.K
        if self.params.pi.shape != (K,):
            raise ValueError("pi must be shape (K,).")
        if self.params.A.shape != (K, K):
            raise ValueError("A must be shape (K,K).")
        if self.cfg.forbid_self_transitions:
            np.fill_diagonal(self.params.A, 0.0)
        # renormalize rows
        self.params.A = (self.params.A.T / np.clip(self.params.A.sum(axis=1), 1e-12, None)).T
        self.params.pi = self.params.pi / np.clip(self.params.pi.sum(), 1e-12, None)

        if self.cfg.min_duration < 1 or self.cfg.Dmax < self.cfg.min_duration:
            raise ValueError("Require 1 <= min_duration <= Dmax.")

        kind, obj = self.params.duration
        if kind == "poisson":
            pd: PoissonDur = obj  # type: ignore[assignment]
            if pd.lam.shape != (K,):
                raise ValueError("PoissonDur.lam must be (K,).")
            if np.any(pd.lam <= 0):
                raise ValueError("Poisson λ must be > 0.")
        elif kind == "negbin":
            nb: NegBinDur = obj  # type: ignore[assignment]
            if nb.r.shape != (K,) or nb.p.shape != (K,):
                raise ValueError("NegBinDur r and p must be (K,).")
            if np.any(nb.r <= 0) or np.any((nb.p <= 0) | (nb.p >= 1)):
                raise ValueError("NegBin parameters invalid.")
        else:
            raise ValueError("duration kind must be 'poisson' or 'negbin'.")

    @staticmethod
    def _check_loglik(loglik_tk: ArrayF) -> ArrayF:
        L = np.asarray(loglik_tk, dtype=float)
        if L.ndim != 2:
            raise ValueError("loglik_tk must be (T, K).")
        if not np.all(np.isfinite(L)):
            raise ValueError("loglik_tk contains non-finite values.")
        return L

    # ----------- duration log pmf table (per state) -----------

    def _log_dur_table(self, T: int) -> ArrayF:
        """
        Build a (K, D) table of log duration pmfs over d∈[1..Dcap],
        where Dcap = min(Dmax, T) and enforcing min_duration by masking.
        """
        if T < self.cfg.min_duration:
            raise ValueError(
                f"T={T} is smaller than min_duration={self.cfg.min_duration}"
            )
        if T in self._dur_cache:
            return self._dur_cache[T]
        K = self.cfg.K
        Dcap = min(self.cfg.Dmax, T)
        d_vals = np.arange(1, Dcap + 1, dtype=float)

        kind, obj = self.params.duration
        if kind == "poisson":
            pd: PoissonDur = obj  # type: ignore[assignment]
            lam = pd.lam[:, None]
            base = -lam + d_vals * np.log(lam) - _log_factorial(d_vals.astype(int))[None, :]
            logp = base - logsumexp(base, axis=1)[:, None]
        else:
            nb: NegBinDur = obj  # type: ignore[assignment]
            r = nb.r[:, None]
            p = nb.p[:, None]
            ds = d_vals[None, :]
            log_all = (
                np.vectorize(math.lgamma)(ds + r)
                - np.vectorize(math.lgamma)(r)
                - _log_factorial(ds.astype(int))
                + ds * np.log(p)
                + r * np.log(1.0 - p)
            )
            logp = log_all - logsumexp(log_all, axis=1)[:, None]

        # mask durations < min_duration
        if self.cfg.min_duration > 1:
            mask = d_vals < self.cfg.min_duration
            logp[:, mask] = LOGZERO
            norm = logsumexp(logp, axis=1)
            logp = logp - norm[:, None]

        self._dur_cache[T] = logp
        return logp  # shape (K, Dcap)

    # ----------- segment emission log-likelihoods via cumulative sums -----------

    @staticmethod
    def _segment_ll_from_pointwise(L: ArrayF) -> Tuple[ArrayF, ArrayF]:
        """
        Build cumulative sums to query segment emission log-lik quickly.
        Return (cum, cum_shift) s.t. segLL(j, t, d) = cum[t, j] - cum[t-d, j].
        """
        T, K = L.shape
        cum = np.zeros((T + 1, K), dtype=float)
        np.cumsum(L, axis=0, out=cum[1:])
        return cum, cum  # alias for clarity

    # ----------- E-step: HSMM forward-backward (log-space) -----------

    def _e_step(self, loglik_tk: ArrayF) -> HSMMSufficient:
        T, K = loglik_tk.shape
        Dcap = min(self.cfg.Dmax, T)
        log_dur = self._log_dur_table(T)  # (K, Dcap)
        cum, _ = self._segment_ll_from_pointwise(loglik_tk)

        # alpha[t, j] = log p(y_{1..t}, state j ends at t)
        log_alpha = np.full((T + 1, K), LOGZERO, dtype=float)
        log_alpha[0, :] = LOGZERO  # no segment ended at t=0
        # helper: logphi[u, j] = log sum_i alpha[u, i] + log A[i,j]
        logphi = np.full((T + 1, K), LOGZERO, dtype=float)
        logphi[0, :] = LOGZERO

        log_pi = safe_log(self.params.pi)
        logA = safe_log(self.params.A)

        for t in range(1, T + 1):
            dmax_t = min(Dcap, t)
            u = t - np.arange(1, dmax_t + 1)
            needed = u[(u > 0) & (logphi[u, 0] <= LOGZERO / 2)]
            for uu in np.unique(needed):
                logphi[uu, :] = logsumexp(log_alpha[uu, :, None] + logA, axis=0)
            seg_ll = cum[t, :, None] - cum[u, :].T  # (K, dmax_t)
            trans = np.where(u == 0, log_pi[:, None], logphi[u, :].T)
            terms = trans + log_dur[:, :dmax_t] + seg_ll
            log_alpha[t, :] = logsumexp(terms, axis=1)

        logZ = _as_scalar(logsumexp(log_alpha[T, :]))  # sequence log-likelihood

        # Backward β[t, j] = log p(y_{t+1..T} | last ended at t, and was j)
        log_beta = np.full((T + 1, K), LOGZERO, dtype=float)
        log_beta[T, :] = 0.0

        # helper g[t, m] = log sum_{d'} p_dur[m,d'] * segLL(m, t+1..t+d') + β[t+d', m]
        g = np.full((T + 1, K), LOGZERO, dtype=float)
        g[T, :] = LOGZERO  # no segment can start after T

        for t in range(T - 1, -1, -1):
            dmax_t = min(Dcap, T - t)
            u = t + np.arange(1, dmax_t + 1)
            seg_ll = cum[u, :].T - cum[t, :][:, None]  # (K, dmax_t)
            terms = log_dur[:, :dmax_t] + seg_ll + log_beta[u, :].T
            g[t, :] = logsumexp(terms, axis=1)
            log_beta[t, :] = logsumexp(logA + g[t, :], axis=1)

        # Posterior over segment ends: eta[t, j, d]
        eta = np.zeros((T + 1, K, Dcap), dtype=float)
        for t in range(1, T + 1):
            dmax_t = min(Dcap, t)
            u = t - np.arange(1, dmax_t + 1)
            needed = u[(u > 0) & (logphi[u, 0] <= LOGZERO / 2)]
            for uu in np.unique(needed):
                logphi[uu, :] = logsumexp(log_alpha[uu, :, None] + logA, axis=0)
            seg_ll = cum[t, :, None] - cum[u, :].T
            trans = np.where(u == 0, log_pi[:, None], logphi[u, :].T)
            num = trans + log_dur[:, :dmax_t] + seg_ll + log_beta[t, :][:, None]
            eta[t, :, :dmax_t] = np.exp(num - logZ)

        # Aggregate sufficient stats
        seg_count = eta[1:].sum(axis=(0, 2))                            # (K,)
        durations = np.arange(1, Dcap + 1, dtype=float)
        seg_total_dur = (eta[1:] * durations[None, None, :]).sum(axis=(0, 2))
        seg_total_d2 = (eta[1:] * (durations[None, None, :] ** 2)).sum(axis=(0, 2))

        # Initial segment counts (t == d)
        pi_counts = np.zeros(K, dtype=float)
        for t in range(1, min(T, Dcap) + 1):
            pi_counts += eta[t, :, t - 1]

        # Transition counts
        xi_counts = np.zeros((K, K), dtype=float)
        for t in range(1, T + 1):
            dmax_t = min(Dcap, t)
            u = t - np.arange(1, dmax_t + 1)
            valid = u > 0
            if not np.any(valid):
                continue
            idx = np.arange(dmax_t)[valid]
            for k_idx, u_k in zip(idx, u[valid]):
                w = log_alpha[u_k, :, None] + logA  # (K,K)
                w = np.exp(w - logsumexp(w, axis=0)[None, :])
                xi_counts += w * eta[t, :, k_idx][None, :]

        # Occupancy γ_t(j): accumulate eta coverage with a difference trick
        gamma = np.zeros((T, K), dtype=float)
        durations = np.arange(1, Dcap + 1)
        for j in range(K):
            diff = np.zeros(T + 1, dtype=float)
            w = eta[1:, j, :]
            t_idx = np.arange(1, T + 1)[:, None]
            u = t_idx - durations[None, :]
            mask = u >= 0
            starts = u[mask].astype(int)
            ends = np.broadcast_to(t_idx, u.shape)[mask].astype(int)
            weights = w[mask]
            np.add.at(diff, starts, weights)
            np.add.at(diff, ends, -weights)
            gamma[:, j] = np.cumsum(diff[:-1])

        return HSMMSufficient(
            eta=eta[1:],  # drop t=0
            seg_count=seg_count,
            seg_total_dur=seg_total_dur,
            seg_total_d2=seg_total_d2,
            pi_counts=pi_counts,
            xi_counts=xi_counts,
            gamma=gamma,
            loglik=logZ,
        )

    # ----------- M-step: durations + transitions -----------

    def _m_step_durations(self, S: HSMMSufficient) -> None:
        kind, obj = self.params.duration
        K = self.cfg.K
        Dcap = min(self.cfg.Dmax, S.gamma.shape[0])
        N_d = S.eta  # (T, K, Dcap) with durations aligned to index = d-1

        if kind == "poisson":
            pd: PoissonDur = obj  # type: ignore[assignment]
            new_lam = pd.lam.copy()
            for j in range(K):
                counts = N_d[:, j, :].sum(axis=0)  # expected counts per duration (1..Dcap)
                lam = _as_scalar(pd.lam[j])
                lam = max(lam, 1e-3)
                Ntot = _as_scalar(counts.sum())
                if Ntot <= 0:
                    continue
                # Newton on truncated Poisson likelihood
                for _ in range(30):
                    # gradient: sum_d N_d (d/λ - 1) - Ntot * Z'/Z
                    d_vals = np.arange(1, Dcap + 1, dtype=float)
                    g_emp = _as_scalar(np.dot(counts, (d_vals / lam) - 1.0))
                    z1, z2 = _poisson_dlogZ_terms(lam, Dcap)
                    grad = g_emp - Ntot * z1
                    # Hessian (negative definite): -sum N_d * d/λ^2 - Ntot * (Z''/Z - (Z'/Z)^2)
                    h_emp = _as_scalar(-np.dot(counts, d_vals / (lam * lam)))
                    h = h_emp - Ntot * (z2 - z1 * z1)
                    if not np.isfinite(grad) or not np.isfinite(h) or h >= 0.0:
                        break
                    step = grad / h
                    # damping
                    step = np.clip(step, -lam * 0.5, lam * 0.5)
                    lam_new = max(lam - step, 1e-6)
                    if abs(lam_new - lam) < 1e-6 * (1.0 + lam):
                        lam = lam_new
                        break
                    lam = lam_new
                new_lam[j] = lam
            self.params.duration = ("poisson", PoissonDur(lam=new_lam))

        else:
            nb: NegBinDur = obj  # type: ignore[assignment]
            r_new = nb.r.copy()
            p_new = nb.p.copy()
            for j in range(K):
                Nseg = _as_scalar(S.seg_count[j])
                if Nseg <= 0:
                    continue
                mu = _as_scalar(S.seg_total_dur[j] / max(Nseg, 1e-12))
                var = _as_scalar(S.seg_total_d2[j] / max(Nseg, 1e-12) - mu * mu)
                var = max(var, 1e-8)
                # NB2 MoM (ignores upper truncation; robust and fast):
                # Var = mu + mu^2 / r  => r = mu^2 / (var - mu)  (clip to min)
                if var > mu + 1e-6:
                    r = mu * mu / (var - mu)
                    r = _as_scalar(np.clip(r, 1e-3, 1e6))
                else:
                    r = 1e6  # approx Poisson when overdispersion tiny
                p = mu / (mu + r)  # in (0,1)
                p = _as_scalar(np.clip(p, 1e-6, 1 - 1e-6))
                r_new[j] = r
                p_new[j] = p
            self.params.duration = ("negbin", NegBinDur(r=r_new, p=p_new))

        # updated duration parameters invalidate cached tables
        self._dur_cache.clear()

    def _m_step_transitions(self, S: HSMMSufficient) -> None:
        # initial
        pi = S.pi_counts.copy()
        if pi.sum() > 0:
            pi /= pi.sum()
        else:
            pi = np.full_like(self.params.pi, 1.0 / self.cfg.K)
        # transitions
        A = S.xi_counts.copy()
        if self.cfg.forbid_self_transitions:
            np.fill_diagonal(A, 0.0)
        row_sums = np.clip(A.sum(axis=1, keepdims=True), 1e-12, None)
        A = A / row_sums
        self.params = HSMMParams(pi=pi, A=A, duration=self.params.duration)

    # ----------- Viterbi (max-product HSMM) -----------

    def _viterbi(self, loglik_tk: ArrayF) -> Tuple[ArrayI, ArrayI]:
        T, K = loglik_tk.shape
        Dcap = min(self.cfg.Dmax, T)
        log_dur = self._log_dur_table(T)   # (K, Dcap)
        cum, _ = self._segment_ll_from_pointwise(loglik_tk)
        logA = safe_log(self.params.A)
        log_pi = safe_log(self.params.pi)

        # V[t, j] = best log-score where a segment in state j ends at t
        V = np.full((T + 1, K), LOGZERO, dtype=float)
        bp_prev_state = np.full((T + 1, K), -1, dtype=int)
        bp_prev_time = np.full((T + 1, K), -1, dtype=int)
        V[0, :] = LOGZERO

        for t in range(1, T + 1):
            for j in range(K):
                best = LOGZERO
                best_i = -1
                best_u = -1
                dmax_t = min(Dcap, t)
                for d in range(1, dmax_t + 1):
                    u = t - d
                    seg_ll = _as_scalar(cum[t, j] - cum[u, j])
                    if u == 0:
                        cand = _as_scalar(log_pi[j]) + _as_scalar(log_dur[j, d - 1]) + seg_ll
                        prev_i = -1
                    else:
                        prev_scores = V[u, :] + logA[:, j]
                        i = int(np.argmax(prev_scores))
                        cand = _as_scalar(prev_scores[i]) + _as_scalar(log_dur[j, d - 1]) + seg_ll
                        prev_i = i
                    if cand > best:
                        best = cand
                        best_i = prev_i
                        best_u = u
                V[t, j] = best
                bp_prev_state[t, j] = best_i
                bp_prev_time[t, j] = best_u

        # backtrack best ending state at T
        jT = int(np.argmax(V[T, :]))
        states = np.empty(T, dtype=int)
        durs = np.zeros(T, dtype=int)  # optional, per-end indicator
        t = T
        while t > 0:
            j = jT
            u = int(bp_prev_time[t, j])
            # fill the segment (u..t]
            states[u:t] = j
            durs[t - 1] = t - u
            jT = int(bp_prev_state[t, j])
            t = u

        return states.astype(int), durs.astype(int)
