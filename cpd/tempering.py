# tempering.py
# MIT License
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

# Types and model hooks
Tau = Tuple[int, ...]


@dataclass
class PTConfig:
    """
    Parallel Tempering configuration for two chains (cold T=1, hot T>1).
    """
    iters: int = 20_000
    burn: int = 10_000
    thin: int = 10
    swap_every: int = 50
    T_hot: float = 3.0
    seed: Optional[int] = 123


@dataclass
class PTResult:
    samples_tau_cold: List[Tau]
    log_posts_cold: List[float]
    cp_hist_cold: NDArray[np.int64]
    mode_tau_cold: Tau
    swaps_attempted: int
    swaps_accepted: int


def _mh_step_with_temperature(model, tau: Tau, T: float) -> tuple[Tau, float]:
    """
    Single Metropolis-Hastings step at temperature T using model's internal proposal mechanisms.
    Reuses the exact logic as in WithinPeriodCPD.fit, but scales acceptance by 1/T.
    """
    N = model._N  # prepared by _prepare_counts
    l = model._l

    def is_valid(t: Tau) -> bool:
        # Reuse internal validator
        from within_period_cpd import _is_valid_tau  # local import to avoid circulars when not needed elsewhere
        return _is_valid_tau(t, N, l)

    m = 1 if len(tau) == 0 else len(tau) + 1
    log_cur = model._log_posterior_tau(tau)

    if m == 1:
        cand_list = model._uniform_birth_targets_m1()
        q_fwd = 1.0 / len(cand_list)
        tau_prop = random.choice(cand_list)
        if tau_prop == ():
            q_bwd = q_fwd
        else:
            death_cands = model._uniform_death_targets(tau_prop)
            q_bwd = 1.0 / len(death_cands)
    else:
        # Use same move/birth/death mix as default RJConfig
        move_prob = 0.5
        birth_prob = 0.25
        death_prob = 0.25
        s = move_prob + birth_prob + death_prob
        move_prob, birth_prob, death_prob = move_prob / s, birth_prob / s, death_prob / s
        u = random.random()

        if u < move_prob:
            j = random.randrange(len(tau))
            cand_list = model._uniform_move_targets(tau, j)
            tau_prop = random.choice(cand_list)
            q_fwd = move_prob * (1.0 / len(tau)) * (1.0 / len(cand_list))
            if tau_prop == tau:
                q_bwd = q_fwd
            else:
                # approximate symmetric count
                cand_back = model._uniform_move_targets(tau_prop, j if j < len(tau_prop) else 0)
                q_bwd = move_prob * (1.0 / len(tau_prop)) * (1.0 / len(cand_back))

        elif u < move_prob + birth_prob:
            seg_idx = random.randrange(m)
            cand_list = model._uniform_birth_targets(tau, seg_idx)
            tau_prop = random.choice(cand_list)
            q_fwd = birth_prob * (1.0 / m) * (1.0 / len(cand_list))
            if tau_prop == tau:
                q_bwd = q_fwd
            else:
                death_cands = model._uniform_death_targets(tau_prop)
                q_bwd = death_prob * (1.0 / len(tau_prop)) * (1.0 / len(death_cands))

        else:
            cand_list = model._uniform_death_targets(tau)
            tau_prop = random.choice(cand_list)
            q_fwd = death_prob * (1.0 / len(tau)) * (1.0 / len(cand_list))
            if tau_prop == tau:
                q_bwd = q_fwd
            else:
                # approximate symmetric birth-proposal normalization
                m_prop = len(tau_prop) + 1
                total_birth = 0
                for sidx in range(m_prop):
                    total_birth += len(model._uniform_birth_targets(tau_prop, sidx))
                q_bwd = birth_prob / max(1, total_birth)

    log_prop = model._log_posterior_tau(tau_prop)
    log_alpha = ((log_prop - log_cur) / max(1e-12, T)) + math.log(q_bwd) - math.log(q_fwd)
    if math.log(random.random()) < min(0.0, log_alpha):
        return tau_prop, log_prop
    return tau, log_cur


def parallel_tempering_fit(model, x: Sequence[int | bool], ptcfg: PTConfig) -> PTResult:
    """
    Run two-chain Parallel Tempering: cold T=1 and hot T=ptcfg.T_hot, with periodic swaps.
    Keeps and returns ONLY the cold chain's samples (typical practice).
    """
    if ptcfg.seed is not None:
        np.random.seed(ptcfg.seed)
        random.seed(ptcfg.seed)

    x_arr = np.asarray(x, dtype=bool)
    if x_arr.ndim != 1 or x_arr.size < model._N:
        raise ValueError(f"x must be 1-D and length >= N={model._N}.")

    # Prepare counts for likelihood calls
    model._prepare_counts(x_arr)

    # Initialize states
    tau_cold: Tau = ()
    tau_hot: Tau = ()
    log_cold = model._log_posterior_tau(tau_cold)
    log_hot = model._log_posterior_tau(tau_hot)

    kept_taus: List[Tau] = []
    kept_logs: List[float] = []
    cp_hist = np.zeros(model._N, dtype=np.int64)

    swaps_attempted = swaps_accepted = 0

    for it in range(ptcfg.iters):
        # One local MH step per chain
        tau_cold, log_cold = _mh_step_with_temperature(model, tau_cold, T=1.0)
        tau_hot,  log_hot  = _mh_step_with_temperature(model, tau_hot,  T=ptcfg.T_hot)

        # Attempt swap periodically
        if (it + 1) % ptcfg.swap_every == 0:
            swaps_attempted += 1
            # swap acceptance: min(1, exp((1/T1 - 1/T2)*(log_post_hot - log_post_cold)))
            delta = (1.0 - 1.0 / ptcfg.T_hot) * (log_hot - log_cold)
            if math.log(random.random()) < min(0.0, delta):
                swaps_accepted += 1
                tau_cold, tau_hot = tau_hot, tau_cold
                log_cold, log_hot = log_hot, log_cold

        # Record cold chain
        if it >= ptcfg.burn and ((it - ptcfg.burn) % ptcfg.thin == 0):
            kept_taus.append(tau_cold)
            kept_logs.append(log_cold)
            for cp in tau_cold:
                cp_hist[cp] += 1

    # MAP from cold kept samples
    if kept_logs:
        mode_tau = kept_taus[int(np.argmax(kept_logs))]
    else:
        mode_tau = tau_cold

    return PTResult(
        samples_tau_cold=kept_taus,
        log_posts_cold=kept_logs,
        cp_hist_cold=cp_hist,
        mode_tau_cold=mode_tau,
        swaps_attempted=swaps_attempted,
        swaps_accepted=swaps_accepted,
    )

# from tempering import PTConfig, parallel_tempering_fit
# ptres = parallel_tempering_fit(model, x, PTConfig(iters=20000, burn=10000, thin=10, T_hot=3.0, swap_every=50))
# # Use ptres.samples_tau_cold etc.
