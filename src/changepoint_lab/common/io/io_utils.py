# io_utils.py
# MIT License
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

# Local types shared with within-period CPD modules
Tau = Tuple[int, ...]


def _pack_tau_list(samples: Sequence[Tau]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Pack a list of variable-length integer tuples into (flat, idx).
    - flat: concatenated ints
    - idx : start index of each tuple in 'flat' (length = n_samples + 1; last is sentinel = flat.size)
    """
    lengths = np.fromiter((len(t) for t in samples), count=len(samples), dtype=np.int64)
    total = int(lengths.sum())
    flat = np.empty(total, dtype=np.int64)
    idx = np.empty(len(samples) + 1, dtype=np.int64)
    pos = 0
    idx[0] = 0
    for i, t in enumerate(samples):
        if t:
            flat[pos:pos + len(t)] = np.asarray(t, dtype=np.int64)
        pos += len(t)
        idx[i + 1] = pos
    return flat, idx


def _unpack_tau_list(flat: NDArray[np.int64], idx: NDArray[np.int64]) -> List[Tau]:
    """Inverse of _pack_tau_list."""
    out: List[Tau] = []
    for i in range(idx.size - 1):
        a, b = int(idx[i]), int(idx[i + 1])
        out.append(tuple([] if b == a else flat[a:b].tolist()))
    return out


def save_result_npz(
    path: str | Path,
    *,
    samples_tau: Sequence[Tau],
    log_posteriors: Sequence[float],
    changepoint_hist: NDArray[np.integer],
    mode_tau: Tau,
    prior_obj,  # ModelPrior
    cfg_obj,    # RJConfig
) -> None:
    """
    Persist an MCMC run to a compressed NPZ. Stores sampler/prior fields for provenance.
    """
    p = Path(path)
    flat, idx = _pack_tau_list(samples_tau)
    np.savez_compressed(
        p,
        samples_flat=flat,
        samples_idx=idx,
        log_posteriors=np.asarray(log_posteriors, dtype=float),
        cp_hist=np.asarray(changepoint_hist, dtype=np.int64),
        mode_tau=np.asarray(mode_tau, dtype=np.int64),
        # prior
        prior_N=int(prior_obj.N),
        prior_l=int(prior_obj.l),
        prior_gamma=float(prior_obj.gamma),
        prior_pois_lambda=float(prior_obj.pois_lambda),
        # cfg
        cfg_iters=int(cfg_obj.iters),
        cfg_burn=int(cfg_obj.burn),
        cfg_thin=int(cfg_obj.thin),
        cfg_seed=-1 if cfg_obj.seed is None else int(cfg_obj.seed),
        cfg_move_prob=float(cfg_obj.move_prob),
        cfg_birth_prob=float(cfg_obj.birth_prob),
        cfg_death_prob=float(cfg_obj.death_prob),
    )


def load_result_npz(path: str | Path):
    """
    Load a persisted run.
    Returns:
      dict with keys:
        samples_tau (List[Tau]), log_posteriors (np.ndarray), changepoint_hist (np.ndarray),
        mode_tau (Tau), prior (dict), cfg (dict)
    """
    p = Path(path)
    z = np.load(p, allow_pickle=False)
    samples_tau = _unpack_tau_list(z["samples_flat"], z["samples_idx"])
    log_post = z["log_posteriors"].astype(float)
    cp_hist = z["cp_hist"].astype(np.int64)
    mode_tau = tuple(z["mode_tau"].astype(int).tolist())

    prior = {
        "N": int(z["prior_N"]),
        "l": int(z["prior_l"]),
        "gamma": float(z["prior_gamma"]),
        "pois_lambda": float(z["prior_pois_lambda"]),
    }
    cfg = {
        "iters": int(z["cfg_iters"]),
        "burn": int(z["cfg_burn"]),
        "thin": int(z["cfg_thin"]),
        "seed": None if int(z["cfg_seed"]) < 0 else int(z["cfg_seed"]),
        "move_prob": float(z["cfg_move_prob"]),
        "birth_prob": float(z["cfg_birth_prob"]),
        "death_prob": float(z["cfg_death_prob"]),
    }

    return {
        "samples_tau": samples_tau,
        "log_posteriors": log_post,
        "changepoint_hist": cp_hist,
        "mode_tau": mode_tau,
        "prior": prior,
        "cfg": cfg,
    }


# from io_utils import save_result_npz, load_result_npz
# save_result_npz("run.npz",
#     samples_tau=result.samples_tau,
#     log_posteriors=result.log_posteriors,
#     changepoint_hist=result.changepoint_hist,
#     mode_tau=result.mode_tau,
#     prior_obj=prior, cfg_obj=cfg)

# loaded = load_result_npz("run.npz")
