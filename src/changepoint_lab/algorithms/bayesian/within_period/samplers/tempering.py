# tempering.py
# MIT License
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .....core.random import spawn_rngs

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
    provenance: dict[str, object] = field(default_factory=dict)


def _mh_step_with_temperature(
    model,
    tau: Tau,
    T: float,
    rng: np.random.Generator,
) -> tuple[Tau, float]:
    """
    Single Metropolis-Hastings step at temperature T using model's internal proposal mechanisms.
    Reuses the exact logic as in WithinPeriodCPD.fit, but scales acceptance by 1/T.
    """
    from ..within_period_cpd import RJConfig

    cfg = RJConfig(seed=None)
    log_cur = model._log_posterior_tau(tau)
    step = model._sample_proposal(tau, cfg, rng)
    tau_prop = step.tau
    q_fwd = step.probability
    q_bwd = model.proposal_probability(tau_prop, tau, cfg)

    log_prop = model._log_posterior_tau(tau_prop)
    log_alpha = ((log_prop - log_cur) / max(1e-12, T)) + math.log(q_bwd) - math.log(q_fwd)
    if step.kind != "stay" and math.log(float(rng.random())) < min(0.0, log_alpha):
        return tau_prop, log_prop
    return tau, log_cur


def parallel_tempering_fit(model, x: Sequence[int | bool], ptcfg: PTConfig) -> PTResult:
    """
    Run two-chain Parallel Tempering: cold T=1 and hot T=ptcfg.T_hot, with periodic swaps.
    Keeps and returns ONLY the cold chain's samples (typical practice).
    """
    rng_cold, rng_hot, rng_swap = spawn_rngs(ptcfg.seed, 3)

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
        tau_cold, log_cold = _mh_step_with_temperature(model, tau_cold, T=1.0, rng=rng_cold)
        tau_hot, log_hot = _mh_step_with_temperature(
            model, tau_hot, T=ptcfg.T_hot, rng=rng_hot
        )

        # Attempt swap periodically
        if (it + 1) % ptcfg.swap_every == 0:
            swaps_attempted += 1
            # swap acceptance: min(1, exp((1/T1 - 1/T2)*(log_post_hot - log_post_cold)))
            delta = (1.0 - 1.0 / ptcfg.T_hot) * (log_hot - log_cold)
            if math.log(float(rng_swap.random())) < min(0.0, delta):
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
        provenance={
            "seed": ptcfg.seed,
            "rng": "numpy.random.Generator",
            "stream_policy": "SeedSequence.spawn(cold, hot, swap)",
            "iters": ptcfg.iters,
            "burn": ptcfg.burn,
            "thin": ptcfg.thin,
            "swap_every": ptcfg.swap_every,
            "T_hot": ptcfg.T_hot,
        },
    )

# from tempering import PTConfig, parallel_tempering_fit
# ptres = parallel_tempering_fit(model, x, PTConfig(iters=20000, burn=10000, thin=10, T_hot=3.0, swap_every=50))
# # Use ptres.samples_tau_cold etc.
