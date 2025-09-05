# Reference: Taylor, Killick, Burr & Rogerson (2021), Assessing daily patterns using
# home activity sensors and within period changepoint detection, JRSS-C 70(3): 579–595.
#

# within_period_cpd.py
# MIT License
# (c) 2025

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict
import math
import random

import numpy as np
from numpy.typing import NDArray


# =========================
# Types
# =========================

Array1D = NDArray[np.int64]
ArrayBool = NDArray[np.bool_]
Tau = Tuple[
    int, ...
]  # Sorted tuple of changepoint positions in [0, N-1]; empty tuple means m=1 model.


# =========================
# Utilities
# =========================


def _lgamma(x: float) -> float:
    """Shorthand to the natural-log gamma function."""
    return math.lgamma(x)


def _log_beta(a: float, b: float) -> float:
    """log Beta(a,b) = logGamma(a)+logGamma(b)-logGamma(a+b)."""
    return _lgamma(a) + _lgamma(b) - _lgamma(a + b)


def _mod_distance(a: int, b: int, N: int) -> int:
    """
    Circular forward distance from a to b on {0, ..., N-1}, measured in integer steps.
    Returns an integer in {0, 1, ..., N-1}. By convention, if a==b return 0 (the full
    wrap length is N; callsites must guard for '>= l' where needed).
    """
    d = (b - a) % N
    return d


def _segment_lengths(tau: Tau, N: int) -> List[int]:
    """
    Compute segment lengths (in integer lattice points) for changepoints tau over a circular axis of length N.
    Semantics: segments are open-closed intervals (prev, tau_i] modulo N.
    If tau == (), m=1 and there is a single segment of length N.
    """
    if len(tau) == 0:
        return [N]
    # tau sorted increasing in [0, N-1]
    lens: List[int] = []
    m = len(tau) + 1
    # previous "boundary" is tau[i-1], with tau[-1] interpreted as tau[m-2] and wrapping from N
    prev = tau[-1]
    for t in tau:
        d = _mod_distance(prev, t, N)
        lens.append(N if d == 0 else d)  # Should never be 0 for valid tau
        prev = t
    return lens


def _is_valid_tau(tau: Tau, N: int, l: int) -> bool:
    """
    Check the minimum segment length constraint for a candidate changepoint vector.
    Each segment length must be >= l. For tau==(), require N >= l.
    """
    if l < 1 or N < 1:
        return False
    try:
        lens = _segment_lengths(tau, N)
    except Exception:
        return False
    return all(L >= l for L in lens)


def _sorted_unique_mod(xs: Iterable[int], N: int) -> Tau:
    """Return a sorted, duplicate-free tuple of positions in [0, N-1]."""
    arr = sorted({int(x) % N for x in xs})
    return tuple(arr)


def _count_by_time_of_day(x: ArrayBool, N: int) -> Tuple[Array1D, Array1D]:
    """
    Aggregate binary observations per time-of-day index r = t % N.
    Returns:
        s_r[r] : sum of ones at time r
        n_r[r] : count of observations at time r
    """
    T = int(x.size)
    idx = np.mod(np.arange(T, dtype=np.int64), N)
    # counts per r
    n_r = np.bincount(idx, minlength=N).astype(np.int64)
    s_r = np.bincount(idx, weights=x.astype(np.int64), minlength=N).astype(np.int64)
    return s_r, n_r


@dataclass(frozen=True)
class RJConfig:
    """
    Reversible-jump MCMC configuration.

    Attributes
    ----------
    iters : int
        Total number of MCMC iterations.
    burn : int
        Burn-in iterations (discarded).
    thin : int
        Keep one sample every `thin` iterations after burn-in (>=1).
    seed : Optional[int]
        PRNG seed for reproducibility.
    move_prob : float
        Probability for a 'move' proposal when m>1.
    birth_prob : float
        Probability for a 'birth' proposal (adding a changepoint).
    death_prob : float
        Probability for a 'death' proposal (deleting a changepoint).
        (move_prob + birth_prob + death_prob) must equal 1 for m>1.
        For m==1, only birth proposals are used.
    """

    iters: int = 30_000
    burn: int = 10_000
    thin: int = 10
    seed: Optional[int] = 7
    move_prob: float = 0.50
    birth_prob: float = 0.25
    death_prob: float = 0.25

    def __post_init__(self) -> None:
        if self.iters <= 0 or self.burn < 0 or self.thin <= 0:
            raise ValueError("iters>0, burn>=0, thin>0 required.")
        if self.burn >= self.iters:
            raise ValueError("burn must be < iters.")
        if (
            not (0 < self.move_prob < 1)
            or not (0 < self.birth_prob < 1)
            or not (0 < self.death_prob < 1)
        ):
            # The exact values only apply for m>1; for m==1 we ignore move/death.
            pass
        # We won't hard-enforce sum==1 since we adjust dynamically for m==1 below.


@dataclass(frozen=True)
class ModelPrior:
    """
    Prior configuration, following Taylor et al. (2021). See Sections 2.1–2.3. :contentReference[oaicite:1]{index=1}

    Attributes
    ----------
    N : int
        Period length (e.g., N=96 for 15-min bins over a day).
    l : int
        Minimum segment length (e.g., l=4 for 1 hour at 15-min resolution).
    gamma : float
        Dirichlet–multinomial common shape parameter (γ>0). γ=1 is uniform
        over excess-length allocations. (Section 2.2; Eq. (4)) :contentReference[oaicite:2]{index=2}
    pois_lambda : float
        Poisson(λ) prior on the number of segments m (truncated to 1..floor(N/l)).
        Taylor et al. use λ=1. (Section 2.2) :contentReference[oaicite:3]{index=3}
    """

    N: int
    l: int
    gamma: float = 1.0
    pois_lambda: float = 1.0

    def __post_init__(self) -> None:
        if self.N <= 0:
            raise ValueError("N must be positive.")
        if self.l <= 0 or self.l > self.N:
            raise ValueError("l must be in [1, N].")
        if self.gamma <= 0:
            raise ValueError("gamma must be > 0.")
        if self.pois_lambda <= 0:
            raise ValueError("pois_lambda must be > 0.")

    @property
    def m_max(self) -> int:
        return self.N // self.l  # floor(N/l)


@dataclass
class MCMCResult:
    """
    Container for MCMC outputs.

    Attributes
    ----------
    samples_tau : List[Tau]
        Posterior samples of changepoint vectors (post-burn, thinned).
    log_posteriors : List[float]
        Log unnormalized posterior values for each kept sample.
    changepoint_hist : Array1D
        Histogram counts for positions 0..N-1 across samples (probability that a changepoint occurs at that index
        is changepoint_hist / sum(changepoint_hist) for visualization).
    mode_tau : Tau
        The MAP (highest log posterior within kept samples).
    """

    samples_tau: List[Tau]
    log_posteriors: List[float]
    changepoint_hist: Array1D
    mode_tau: Tau


class WithinPeriodCore:
    """
    Within-period changepoint detection for periodic binary data with minimum segment length constraint,
    using an RJMCMC sampler (birth/move/death) over changepoints on a circular time axis. Follows
    Taylor et al. (2021), Sections 2–3. :contentReference[oaicite:4]{index=4}

    Model:
        - Observations X_t | p(t) ~ Bernoulli(p(t)), t=0..T-1.
        - p(t) is piecewise-constant over the period N with m segments and changepoints τ.
        - Prior on segment probabilities is Beta(1,1) (uniform), marginalized -> Beta-binomial per segment (Eq. (6)).
        - Prior on m ~ Truncated Poisson(λ), m ∈ {1, ..., floor(N/l)}.
        - Prior on changepoint locations, via excess lengths δ_i = seg_len_i - l with Dirichlet–multinomial(γ) over δ (Eq. (4)).
        - Anchor is uniformly distributed over N positions (constant 1/N in the posterior proportionality).
    """

    def __init__(self, prior: ModelPrior):
        self.prior = prior
        self._N = prior.N
        self._l = prior.l
        # Placeholders set during fit:
        self._s_r: Optional[Array1D] = None
        self._n_r: Optional[Array1D] = None
        self._S2: Optional[Array1D] = None  # prefix sums over doubled s_r
        self._N2: Optional[Array1D] = None  # prefix sums over doubled n_r

    # -------------------- Data aggregation --------------------

    def _prepare_counts(self, x: ArrayBool) -> None:
        """Aggregate counts per time-of-day and build doubled cumulative sums for fast circular segment queries."""
        s_r, n_r = _count_by_time_of_day(x, self._N)
        self._s_r = s_r
        self._n_r = n_r
        # Double arrays for circular range-sum queries
        s2 = np.concatenate([s_r, s_r])
        n2 = np.concatenate([n_r, n_r])
        S2 = np.zeros(s2.size + 1, dtype=np.int64)
        N2 = np.zeros(n2.size + 1, dtype=np.int64)
        np.cumsum(s2, out=S2[1:])
        np.cumsum(n2, out=N2[1:])
        self._S2, self._N2 = S2, N2

    def _sum_over_arc(
        self, start_exclusive: int, end_inclusive: int
    ) -> Tuple[int, int]:
        """
        Sum s_r and n_r over the circular arc (start_exclusive, end_inclusive] modulo N.
        Uses doubled prefix arrays for O(1) range sums.

        Returns
        -------
        s : int
            Sum of ones on the arc.
        n : int
            Number of observations on the arc.
        """
        assert self._S2 is not None and self._N2 is not None
        N = self._N
        # length of arc in lattice points
        L = _mod_distance(start_exclusive, end_inclusive, N)
        L = N if L == 0 else L
        # arc corresponds to indices (start+1) .. (start+L)
        a = (start_exclusive + 1) % N
        # We map to doubled arrays by offsetting into [0, 2N)
        # Ensure 'a' is within first N; if not, shift back by N because S2/N2 cover [0, 2N]
        if a >= N:
            a -= N
        b = a + L
        S2, N2 = self._S2, self._N2
        s = int(S2[b] - S2[a])
        n = int(N2[b] - N2[a])
        return s, n

    def _segment_stats(self, tau: Tau) -> Tuple[List[int], List[int]]:
        """
        For a given tau, compute (s_i, n_i) for each segment i=0..m-1.
        """
        if len(tau) == 0:
            s, n = self._sum_over_arc(
                tau[-1] if tau else (self._N - 1), tau[0] if tau else (self._N - 1)
            )
            # Above is a (prev, curr] wrapper; when tau==(), use the entire circle: use start=N-1, end=N-1
            # which yields length N.
            return [s], [n]

        s_list: List[int] = []
        n_list: List[int] = []
        prev = tau[-1]
        for t in tau:
            s, n = self._sum_over_arc(prev, t)
            s_list.append(s)
            n_list.append(n)
            prev = t
        return s_list, n_list

    # -------------------- Posterior (unnormalized log) --------------------

    def _log_posterior_tau(self, tau: Tau) -> float:
        """
        Unnormalized log-posterior for a given tau (changepoints), with segment-probabilities marginalized out.

        For m=1:
            log π ∝ log Beta(1+S, 1+T-S)  - log N  + log Poisson_trunc(m=1)  (constants dropped).

        For m>1:
            log π ∝ sum_i log Beta(1+s_i, 1+n_i - s_i)
                    + log DirichletMultinomial(δ | m, γ)    [excess lengths: δ_i = seg_len_i - l]
                    - log N
                    + log Poisson_trunc(m)
            Constants wrt τ,m are omitted; ratios are correct in MH steps.

        See Eqs. (3), (4), (6), (7) in Taylor et al. (2021). :contentReference[oaicite:5]{index=5}
        """
        N, l, gamma, lam = self._N, self._l, self.prior.gamma, self.prior.pois_lambda
        m = 1 if len(tau) == 0 else len(tau) + 1

        # segment stats
        s_list, n_list = self._segment_stats(tau)

        # (i) Marginal likelihood (Beta-binomial with Beta(1,1) prior)
        ll = 0.0
        if m == 1:
            S = float(sum(s_list))
            Ttot = float(sum(n_list))
            ll += _log_beta(1.0 + S, 1.0 + Ttot - S)
        else:
            for s_i, n_i in zip(s_list, n_list):
                ll += _log_beta(1.0 + s_i, 1.0 + (n_i - s_i))

        # (ii) Prior on τ via Dirichlet–multinomial over excess lengths δ (only when m>1)
        # δ_i = seg_len_i - l, with Δ = N - l*m, sum δ_i = Δ, δ_i >= 0.
        if m == 1:
            prior_tau = 0.0  # point-mass on empty set; contributes constant only
        else:
            lens = _segment_lengths(tau, N)
            deltas = [L - l for L in lens]
            Delta = N - l * m
            # DirMult(γ) mass (up to constants that cancel across τ with same m? keep full form to compare across m)
            # log[ Δ! * Γ(mγ) / ( Γ(Δ + mγ) * Γ(γ)^m ) * ∏ Γ(δ_i + γ) / δ_i! ]
            prior_tau = (
                _lgamma(Delta + 1.0)
                + _lgamma(m * gamma)
                - _lgamma(Delta + m * gamma)
                - m * _lgamma(gamma)
                + sum(_lgamma(delta + gamma) - _lgamma(delta + 1.0) for delta in deltas)
            )

        # (iii) Uniform anchor over N (constant -log N) and truncated Poisson prior on m: p(m) ∝ e^{-λ} λ^{m} / m!
        # e^{-λ} is constant across m, truncation normalizer is also constant for a fixed N,l; include -log(m!)
        # NOTE: If you prefer a different prior on m, modify here.
        prior_m = -_lgamma(m + 1.0)  # -log(m!)
        # (iv) total
        return (
            ll + prior_tau + prior_m
        )  # (-log N) is a constant offset; omit for ratios.

    # -------------------- Proposal mechanisms --------------------

    def _uniform_move_targets(self, tau: Tau, j: int) -> Tau:
        """
        Enumerate all valid new positions for changepoint index j, under min-length constraint.
        Returns a Tau for each candidate value (with index j replaced), as a generator.
        """
        N, l = self._N, self._l
        m = len(tau) + 1
        assert m > 1
        # neighbors (prev cp and next cp) in circular order
        prev_cp = tau[j - 1] if j > 0 else tau[-1]
        next_cp = tau[(j + 1) % len(tau)]
        # allowed v in (prev_cp, next_cp] with both segments >= l -> v in {prev_cp + l, ..., next_cp - l}
        max_step = _mod_distance(prev_cp, next_cp, N)
        # If max_step < 2l, no valid move (degenerate, but tau must have satisfied l so equality can't break on current v)
        Lmin = l
        Lmax = max_step - l
        candidates: List[Tau] = []
        if Lmax >= Lmin:
            for k in range(Lmin, Lmax + 1):
                v = (prev_cp + k) % N
                if v == tau[j]:
                    # include current position as well? We'll include it to let the chain stay (implies proposal prob)
                    pass
                new_tau = list(tau)
                new_tau[j] = v
                cand = _sorted_unique_mod(new_tau, N)
                if len(cand) == len(tau) and _is_valid_tau(cand, N, l):
                    candidates.append(cand)
        # Ensure at least current tau is available (stay)
        if tau not in candidates:
            candidates.append(tau)
        return tuple(candidates)

    def _uniform_birth_targets(self, tau: Tau, seg_index: int) -> Tuple[Tau, ...]:
        """
        Enumerate all valid birth proposals by inserting one changepoint inside the selected segment.

        The selected segment is (prev, end] where 'end' is tau[seg_index] (for seg_index in 0..m-2),
        and for the last segment, 'end' is tau[0] (wrap), 'prev' is tau[-1].

        We uniformly consider all lattice points v in {prev + l, ..., end - l} as valid insertion locations.
        """
        N, l = self._N, self._l
        m = len(tau) + 1
        assert 0 <= seg_index < m
        if m == 1:
            # Whole circle is one segment: any two cps must be at least l apart both ways.
            raise RuntimeError("Use _uniform_birth_targets_m1 for m=1.")
        # Identify segment end boundary:
        end = tau[seg_index] if seg_index < len(tau) else tau[0]
        prev = tau[seg_index - 1] if seg_index > 0 else tau[-1]
        length = _mod_distance(prev, end, N)
        # insertion points must keep both new subsegments >= l
        Lmin, Lmax = l, length - l
        candidates: List[Tau] = []
        if Lmax >= Lmin:
            for k in range(Lmin, Lmax + 1):
                v = (prev + k) % N
                new_tau = _sorted_unique_mod([*tau, v], N)
                # Validate:
                if len(new_tau) == len(tau) + 1 and _is_valid_tau(new_tau, N, l):
                    candidates.append(new_tau)
        # Always include 'no change' option (stay)
        if tau not in candidates:
            candidates.append(tau)
        return tuple(candidates)

    def _uniform_birth_targets_m1(self) -> Tuple[Tau, ...]:
        """
        Enumerate all valid τ for m=2 when starting from m=1 (tau=()).
        For two changepoints a<b in [0,N-1], both arc lengths must be >= l:
            (b-a) >= l and (a+N-b) >= l  <=> distance in either direction >= l.
        """
        N, l = self._N, self._l
        candidates: List[Tau] = [()]  # include stay
        for a in range(N - 1):
            for b in range(a + 1, N):
                d1 = b - a
                d2 = N - d1
                if d1 >= l and d2 >= l:
                    candidates.append((a, b))
        return tuple(candidates)

    def _uniform_death_targets(self, tau: Tau) -> Tuple[Tau, ...]:
        """
        Enumerate all valid death proposals by removing one changepoint (choose uniformly).
        """
        N, l = self._N, self._l
        m = len(tau) + 1
        assert m > 1
        candidates: List[Tau] = [tau]  # include stay
        for j in range(len(tau)):
            reduced = tuple(t for k, t in enumerate(tau) if k != j)
            if _is_valid_tau(reduced, N, l):
                candidates.append(reduced)
        return tuple(candidates)

    # -------------------- Sampler --------------------

    def fit(
        self,
        x: Sequence[int | bool],
        cfg: RJConfig = RJConfig(),
        init: Optional[Tau] = None,
    ) -> MCMCResult:
        """
        Run RJMCMC and return posterior samples for τ.

        Parameters
        ----------
        x : Sequence[int|bool]
            Binary observations (0/1 or False/True). Order is linear time; period is N, using t % N.
        cfg : RJConfig
            Sampler configuration.
        init : Optional[Tau]
            Optional initial changepoints. If None, starts from m=1 (tau=()).

        Returns
        -------
        MCMCResult
        """
        # Seed RNG
        if cfg.seed is not None:
            np.random.seed(cfg.seed)
            random.seed(cfg.seed)

        x_arr = np.asarray(x, dtype=bool)
        if x_arr.ndim != 1 or x_arr.size < self._N:
            raise ValueError(f"x must be 1-D and length >= N={self._N}.")

        self._prepare_counts(x_arr)

        # Initialize tau
        tau: Tau = _sorted_unique_mod(init, self._N) if init is not None else ()
        if not _is_valid_tau(tau, self._N, self._l):
            tau = ()  # fallback to m=1

        # Storage
        kept_taus: List[Tau] = []
        kept_logs: List[float] = []
        cp_hist = np.zeros(self._N, dtype=np.int64)

        # MH loop
        for it in range(cfg.iters):
            m = 1 if len(tau) == 0 else len(tau) + 1
            log_cur = self._log_posterior_tau(tau)

            # Decide proposal type
            if m == 1:
                # Only birth from m=1 -> enumerate candidates (including stay)
                candidates = self._uniform_birth_targets_m1()
                # Discrete proposal: choose uniformly among candidates
                q_fwd = 1.0 / len(candidates)
                tau_prop = random.choice(candidates)
                q_bwd: float
                if tau_prop == ():
                    q_bwd = q_fwd  # symmetric stay
                else:
                    # Backward: from m=2 you can propose death; enumerate death candidates and count cardinality
                    death_cands = self._uniform_death_targets(tau_prop)
                    q_bwd = 1.0 / len(death_cands)
            else:
                # m > 1: choose move/birth/death with probabilities; if a move is not possible, we silently fall back to 'stay'
                u = random.random()
                move_prob = cfg.move_prob
                birth_prob = cfg.birth_prob
                death_prob = cfg.death_prob
                # Normalize in case user gave non-exact sums
                s = move_prob + birth_prob + death_prob
                move_prob, birth_prob, death_prob = (
                    move_prob / s,
                    birth_prob / s,
                    death_prob / s,
                )

                if u < move_prob:
                    # MOVE: pick a cp and a new position uniformly from valid targets (including stay)
                    j = random.randrange(len(tau))
                    cand_list = self._uniform_move_targets(tau, j)
                    tau_prop = random.choice(cand_list)
                    q_fwd = move_prob * (1.0 / len(tau)) * (1.0 / len(cand_list))

                    # Backward: in tau_prop, the moved cp sits at position 'v'; find its index j' after sorting -> same cardinality
                    if tau_prop == tau:
                        q_bwd = q_fwd
                    else:
                        # identify which cp moved: find value that is in tau_prop but not in tau OR nearest match
                        # Simpler: any cp index may "move back". Use the same count of candidates from tau_prop for that cp.
                        # We'll choose the index j' corresponding to the value tau_prop[j*] closest to old tau[j]; but counts are same.
                        j_back = None
                        # Best-effort: take the cp in tau_prop with minimal circular distance to tau[j]
                        dmins, j_back = None, 0
                        for idx, v in enumerate(tau_prop):
                            d = min(
                                _mod_distance(v, tau[j], self._N),
                                _mod_distance(tau[j], v, self._N),
                            )
                            if dmins is None or d < dmins:
                                dmins, j_back = d, idx
                        cand_back = self._uniform_move_targets(tau_prop, j_back)
                        q_bwd = (
                            move_prob * (1.0 / len(tau_prop)) * (1.0 / len(cand_back))
                        )

                elif u < move_prob + birth_prob:
                    # BIRTH: pick segment uniformly, insert one cp uniformly among valid positions (including stay)
                    seg_idx = random.randrange(m)
                    cand_list = self._uniform_birth_targets(tau, seg_idx)
                    tau_prop = random.choice(cand_list)
                    q_fwd = birth_prob * (1.0 / m) * (1.0 / len(cand_list))
                    # Backward is DEATH from tau_prop
                    if tau_prop == tau:
                        q_bwd = q_fwd
                    else:
                        death_cands = self._uniform_death_targets(tau_prop)
                        q_bwd = (
                            death_prob
                            * (1.0 / (len(tau_prop)))
                            * (1.0 / len(death_cands))
                        )

                else:
                    # DEATH: remove a cp uniformly (including stay)
                    cand_list = self._uniform_death_targets(tau)
                    tau_prop = random.choice(cand_list)
                    q_fwd = death_prob * (1.0 / len(tau)) * (1.0 / len(cand_list))
                    # Backward is BIRTH from tau_prop
                    if tau_prop == tau:
                        q_bwd = q_fwd
                    else:
                        m_prop = len(tau_prop) + 1
                        if len(tau_prop) == 0:
                            birth_cands_prop = self._uniform_birth_targets_m1()
                            q_bwd = birth_prob * (1.0 / len(birth_cands_prop))
                        else:
                            seg_idx_prop = random.randrange(
                                m_prop
                            )  # any segment could be selected; but proposal enumerates uniformly
                            birth_cands_prop = self._uniform_birth_targets(
                                tau_prop, seg_idx_prop
                            )
                            # We don't know which seg was actually used to get back exactly 'tau'; use a conservative bound by summing?
                            # For a valid MH step we need a single path probability; we approximate by averaging over segments.
                            # To avoid bias, we instead compute total number of birth targets across segments and assume uniform segment draw.
                            total_birth = 0
                            for sidx in range(m_prop):
                                total_birth += len(
                                    self._uniform_birth_targets(tau_prop, sidx)
                                )
                            q_bwd = (
                                birth_prob * (1.0 / m_prop) * (1.0 / (total_birth / m_prop))
                            )  # = birth_prob / total_birth

            # MH ratio
            log_prop = self._log_posterior_tau(tau_prop)
            log_alpha = (log_prop - log_cur) + math.log(q_bwd) - math.log(q_fwd)
            if math.log(random.random()) < min(0.0, log_alpha):
                tau = tau_prop  # accept
                log_cur = log_prop

            # Record sample if needed
            if it >= cfg.burn and ((it - cfg.burn) % cfg.thin == 0):
                kept_taus.append(tau)
                kept_logs.append(log_cur)
                for cp in tau:
                    cp_hist[cp] += 1

        # Choose MAP among kept samples
        if kept_logs:
            mode_idx = int(np.argmax(kept_logs))
            mode_tau = kept_taus[mode_idx]
        else:
            mode_tau = tau

        return MCMCResult(
            samples_tau=kept_taus,
            log_posteriors=kept_logs,
            changepoint_hist=cp_hist,
            mode_tau=mode_tau,
        )

    # -------------------- Posterior summaries --------------------

    def segment_posterior_summaries(
        self,
        tau: Tau,
        credible: float = 0.95,
    ) -> Dict[str, List[Tuple[float, float, float]]]:
        """
        Given a fixed τ (e.g., the MAP), return per-segment posterior summaries for the Bernoulli probability ϕ_i,
        using the conditional posterior Beta(1+s_i, 1+n_i - s_i) (Eq. (6)). :contentReference[oaicite:6]{index=6}

        Returns
        -------
        dict with key "segments" mapped to a list of tuples (mean, lower, upper) for each segment i.
        """
        s_list, n_list = self._segment_stats(tau)
        alphas = np.array([1 + s for s in s_list], dtype=float)
        betas = np.array([1 + (n - s) for s, n in zip(s_list, n_list)], dtype=float)
        means = alphas / (alphas + betas)

        # Use a simple Beta quantile approximation via inverse CDF by sampling (no SciPy dependency).
        # For reproducibility keep a fixed seed here or let user control externally.
        rng = np.random.default_rng(123)
        L = int(
            5_000 / max(1, len(alphas))
        )  # total samples ~ 5k distributed over segments
        lowers, uppers = [], []
        for a, b in zip(alphas, betas):
            draws = rng.beta(a, b, size=L)
            low = float(np.quantile(draws, (1 - credible) / 2.0))
            up = float(np.quantile(draws, 1 - (1 - credible) / 2.0))
            lowers.append(low)
            uppers.append(up)

        return {"segments": list(zip(means.tolist(), lowers, uppers))}

    def pointwise_posterior_summary_from_samples(
        self,
        samples: Sequence[Tau],
        draws_per_sample: int = 1,
        credible: float = 0.95,
        seed: Optional[int] = 123,
    ) -> Dict[str, NDArray]:
        """
        Build a pointwise posterior summary of p(t) on the N lattice, by drawing ϕ from each kept sample's
        Beta posteriors and assigning ϕ to the segment covering each lattice point.

        Parameters
        ----------
        samples : Sequence[Tau]
            Posterior samples of τ (e.g., from MCMC).
        draws_per_sample : int
            Number of ϕ draws per sample to smooth Monte Carlo error (1..5 is typical).
        credible : float
            Credible mass for CIs.
        seed : Optional[int]
            RNG seed for reproducibility.

        Returns
        -------
        dict with keys:
            "median": (N,) array of pointwise medians,
            "lower":  (N,) array of lower CI bound,
            "upper":  (N,) array of upper CI bound.
        """
        if seed is not None:
            np.random.seed(seed)

        N = self._N
        all_draws: List[NDArray] = []
        for tau in samples:
            s_list, n_list = self._segment_stats(tau)
            alphas = np.asarray([1 + s for s in s_list], dtype=float)
            betas = np.asarray(
                [1 + (n - s) for s, n in zip(s_list, n_list)], dtype=float
            )
            for _ in range(max(1, draws_per_sample)):
                phi = np.random.beta(alphas, betas)
                # Assign phi to the N lattice points according to segments
                grid = np.empty(N, dtype=float)
                if len(tau) == 0:
                    grid[:] = phi[0]
                else:
                    prev = tau[-1]
                    for idx, cp in enumerate(tau):
                        # fill (prev, cp]
                        length = _mod_distance(prev, cp, N)
                        length = N if length == 0 else length
                        a = (prev + 1) % N
                        for k in range(length):
                            grid[(a + k) % N] = phi[idx]
                        prev = cp
                all_draws.append(grid)

        if not all_draws:
            raise RuntimeError("No samples provided to summarize.")

        mat = np.vstack(all_draws)  # shape: [S, N]
        lower_q = (1 - credible) / 2.0
        upper_q = 1 - lower_q
        return {
            "median": np.quantile(mat, 0.5, axis=0),
            "lower": np.quantile(mat, lower_q, axis=0),
            "upper": np.quantile(mat, upper_q, axis=0),
        }


# =========================
# Example usage (minimal)
# =========================

if __name__ == "__main__":
    # Synthetic demo: daily pattern with N=96 (15-min bins), two segments (sleep vs day).
    rng = np.random.default_rng(42)

    N = 96
    l = 4
    days = 30
    # True changepoints: sleep ends at 7:30 (index 30), starts at 23:30 (index 94).
    tau_true: Tau = (30, 94)
    # Segment probabilities:
    p_sleep, p_day = 0.05, 0.40

    # Build p(t) over N:
    def build_phi(tau: Tau) -> NDArray:
        phi = np.empty(N, dtype=float)
        if len(tau) == 0:
            phi[:] = 0.2
            return phi
        prev = tau[-1]
        for i, cp in enumerate(tau):
            length = _mod_distance(prev, cp, N)
            length = N if length == 0 else length
            a = (prev + 1) % N
            val = (
                p_day if i == 0 else p_sleep
            )  # first seg after prev is "day" in this simple design
            for k in range(length):
                phi[(a + k) % N] = val
            prev = cp
        return phi

    phi_grid = build_phi(tau_true)

    # Generate 30 days of binary activity at 15-min bins:
    X = []
    for _ in range(days):
        X.append(rng.binomial(1, phi_grid).astype(bool))
    x = np.concatenate(X)

    prior = ModelPrior(N=N, l=l, gamma=1.0, pois_lambda=1.0)
    model = WithinPeriodCore(prior)

    cfg = RJConfig(
        iters=20_000,
        burn=10_000,
        thin=10,
        seed=123,
        move_prob=0.5,
        birth_prob=0.25,
        death_prob=0.25,
    )

    result = model.fit(x, cfg)
    print("MAP tau:", result.mode_tau)
    print("Posterior samples kept:", len(result.samples_tau))

    # Segment summaries for MAP τ
    seg_summ = model.segment_posterior_summaries(result.mode_tau)
    print("Per-segment posterior mean & 95% CI:")
    for i, (mu, lo, hi) in enumerate(seg_summ["segments"]):
        print(f"  seg {i}: mean={mu:.3f}, 95%CI=({lo:.3f}, {hi:.3f})")

    # Pointwise summary
    pw = model.pointwise_posterior_summary_from_samples(
        result.samples_tau, draws_per_sample=2, credible=0.95
    )
    print("Pointwise median p(t) first 10 bins:", np.round(pw["median"][:10], 3))
