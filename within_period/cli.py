# cli.py
# MIT License
# (c) 2025
"""
Short CLI for Within-Period CPD

Features
--------
1) Fit on CSV (no pandas) or run a synthetic demo.
2) Choose RJMCMC (default) or two-chain Parallel Tempering with --pt.
3) Save results to NPZ and export plots.
4) Load an NPZ and re-generate plots with --plot-only.

Examples
--------
# Synthetic demo (RJMCMC), save to ./out
python -m cli --demo --N 96 --l 4 --days 30 --iters 20000 --burn 10000 --thin 10 --outdir out

# Synthetic demo using PT
python -m cli --demo --N 96 --l 4 --days 30 --pt --iters 20000 --burn 10000 --thin 10 --outdir out_pt

# Fit from CSV (timestamp column "ts", 15-min bins, day starts at 00:00)
python -m cli --csv events.csv --timestamp-col ts --bin-minutes 15 --start-hour 0 --l 4 --iters 20000 --burn 10000 --thin 10 --outdir out_csv

# Load a previous run and just plot
python -m cli --load out/run.npz --plot-only --outdir out_plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Local modules
from .within_period_cpd import WithinPeriodCPD, ModelPrior, RJConfig, Tau
from common.plotting.plotting_helpers import (
    plot_changepoint_posterior_mass,
    plot_pointwise_bands,
    plot_posterior_num_segments,
)
from common.diagnostics.diagnostics import posterior_num_segments
from common.io.io_utils import save_result_npz, load_result_npz
from common.io.data_loader import load_binary_from_csv, empirical_per_bin_mean
from .samplers.tempering import PTConfig, parallel_tempering_fit


# --------------------------- Plotting orchestration ---------------------------

def _save_all_plots(
    *,
    prior: ModelPrior,
    samples_tau: list[Tau],
    cp_hist: np.ndarray,
    mode_tau: Tau,
    model: WithinPeriodCPD,
    outdir: Path,
    start_hour: int,
    hours_step: int,
    empirical_mean: Optional[np.ndarray] = None,
    title_suffix: str = "",
) -> None:
    """
    Generate and save:
      - cp_mass.png
      - pointwise_bands.png  (with optional empirical overlay)
      - posterior_m.png
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Pointwise summary from samples
    pw = model.pointwise_posterior_summary_from_samples(
        samples_tau, draws_per_sample=2, credible=0.95
    )

    # 1) Changepoint posterior mass
    plot_changepoint_posterior_mass(
        cp_hist=cp_hist,
        num_samples=len(samples_tau),
        N=prior.N,
        tau_map=mode_tau,
        start_hour=start_hour,
        hours_step=hours_step,
        title=f"Changepoint posterior mass{title_suffix}",
    )
    plt.savefig(outdir / "cp_mass.png", bbox_inches="tight", dpi=150)
    plt.close()

    # 2) Pointwise posterior bands (+ optional empirical overlay)
    ax = plot_pointwise_bands(
        pw=pw,
        tau=mode_tau,
        start_hour=start_hour,
        hours_step=hours_step,
        title=f"Pointwise posterior bands{title_suffix}",
    )
    if empirical_mean is not None and empirical_mean.size == prior.N:
        ax.plot(np.arange(prior.N), empirical_mean, linestyle=":", linewidth=1.2)
    plt.savefig(outdir / "pointwise_bands.png", bbox_inches="tight", dpi=150)
    plt.close()

    # 3) Posterior over m
    pm = posterior_num_segments(samples_tau)
    plot_posterior_num_segments(pm.m_values, pm.probs, title=f"Posterior over m{title_suffix}")
    plt.savefig(outdir / "posterior_m.png", bbox_inches="tight", dpi=150)
    plt.close()

    # Also write a tiny text summary
    (outdir / "summary.txt").write_text(
        f"MAP tau: {mode_tau}\n"
        f"kept samples: {len(samples_tau)}\n"
        f"posterior m: {list(zip(pm.m_values.tolist(), pm.probs.round(3).tolist()))}\n"
    )


# --------------------------- Data helpers ---------------------------

def _build_synth_demo(N: int, l: int, days: int, seed: int) -> np.ndarray:
    """
    Build a simple two-segment daily pattern and simulate 'days' binary days.
    """
    rng = np.random.default_rng(seed)
    # Two cps: day starts at N//4, sleep starts near end
    tau_true: Tau = (N // 4, (N - 2) % N)
    p_lo, p_hi = 0.05, 0.35

    phi = np.full(N, p_hi, dtype=float)
    a = (tau_true[-1] + 1) % N
    length = (tau_true[0] - tau_true[-1]) % N or N
    for k in range(length):
        phi[(a + k) % N] = p_lo

    X = np.concatenate([rng.binomial(1, phi).astype(bool) for _ in range(days)])
    return X


# --------------------------- Fit orchestrators ---------------------------

def _fit_rjmcmc(
    x: np.ndarray,
    prior: ModelPrior,
    *,
    iters: int,
    burn: int,
    thin: int,
    seed: Optional[int],
) -> Tuple[list[Tau], np.ndarray, Tau, list[float]]:
    """
    Fit standard RJMCMC and return (samples_tau, cp_hist, mode_tau, log_posts).
    """
    model = WithinPeriodCPD(prior)
    cfg = RJConfig(iters=iters, burn=burn, thin=thin, seed=seed)
    result = model.fit(x, cfg)
    return result.samples_tau, result.changepoint_hist, result.mode_tau, result.log_posteriors


def _fit_pt(
    x: np.ndarray,
    prior: ModelPrior,
    *,
    iters: int,
    burn: int,
    thin: int,
    seed: Optional[int],
    T_hot: float,
    swap_every: int,
) -> Tuple[list[Tau], np.ndarray, Tau, list[float]]:
    """
    Fit two-chain Parallel Tempering (returning the cold chain’s kept samples and stats).
    """
    model = WithinPeriodCPD(prior)
    ptcfg = PTConfig(iters=iters, burn=burn, thin=thin, seed=seed, T_hot=T_hot, swap_every=swap_every)
    ptres = parallel_tempering_fit(model, x, ptcfg)
    # We return the cold chain outputs in a result-like shape
    return ptres.samples_tau_cold, ptres.cp_hist_cold, ptres.mode_tau_cold, ptres.log_posts_cold


# --------------------------- Main CLI ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Within-period CPD: short CLI")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Run a synthetic demo.")
    mode.add_argument("--csv", type=str, help="Path to CSV with timestamps (use with --timestamp-col).")
    mode.add_argument("--load", type=str, help="Load a previous NPZ result and (re)plot.")

    # CSV options
    ap.add_argument("--timestamp-col", type=str, default="timestamp", help="CSV column with ISO timestamps.")
    ap.add_argument("--value-col", type=str, default=None, help="Optional numeric column to threshold for events.")
    ap.add_argument("--value-threshold", type=float, default=0.0, help="Threshold for value_col to mark an event.")

    # Binning / prior
    ap.add_argument("--bin-minutes", type=int, default=15, help="Minutes per bin (CSV path).")
    ap.add_argument("--start-hour", type=int, default=0, help="Hour that maps to index 0 (CSV path).")
    ap.add_argument("--N", type=int, default=96, help="Bins per day (demo path).")
    ap.add_argument("--l", type=int, default=4, help="Minimum segment length in bins.")

    # Sampler config (used by both RJMCMC and PT)
    ap.add_argument("--iters", type=int, default=20000, help="MCMC iterations.")
    ap.add_argument("--burn", type=int, default=10000, help="Burn-in.")
    ap.add_argument("--thin", type=int, default=10, help="Thinning.")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed.")

    # PT toggles
    ap.add_argument("--pt", action="store_true", help="Use two-chain parallel tempering.")
    ap.add_argument("--T-hot", type=float, default=3.0, help="Hot chain temperature (PT).")
    ap.add_argument("--swap-every", type=int, default=50, help="Swap frequency in iterations (PT).")

    # Demo extras
    ap.add_argument("--days", type=int, default=30, help="Number of synthetic days (demo).")

    # IO
    ap.add_argument("--outdir", type=Path, default=Path("out"), help="Directory for outputs (plots, npz).")
    ap.add_argument("--save-npz", type=str, default="run.npz", help="Filename for saved NPZ (fit paths).")
    ap.add_argument("--plot-only", action="store_true", help="Only plot (used with --load).")
    ap.add_argument("--days-span", type=int, default=None, help="Force an exact # of days when reading CSV.")

    args = ap.parse_args()

    # -------------------- Load-and-plot path --------------------
    if args.load:
        loaded = load_result_npz(args.load)
        prior = ModelPrior(
            N=int(loaded["prior"]["N"]),
            l=int(loaded["prior"]["l"]),
            gamma=float(loaded["prior"]["gamma"]),
            pois_lambda=float(loaded["prior"]["pois_lambda"]),
        )
        # Recreate model with prior to compute pointwise summaries
        model = WithinPeriodCPD(prior)

        samples_tau = list(loaded["samples_tau"])
        cp_hist = loaded["changepoint_hist"]
        mode_tau = tuple(loaded["mode_tau"])  # type: ignore[assignment]

        _save_all_plots(
            prior=prior,
            samples_tau=samples_tau,
            cp_hist=cp_hist,
            mode_tau=mode_tau,
            model=model,
            outdir=args.outdir,
            start_hour=args.start_hour,
            hours_step=6,
            empirical_mean=None,
            title_suffix=" (loaded)",
        )
        print(f"[OK] Plots written to: {args.outdir}")
        return

    # -------------------- Data preparation (demo or csv) --------------------
    if args.demo:
        x = _build_synth_demo(args.N, args.l, args.days, args.seed)
        N = args.N
        empirical = None  # we could compute per-bin mean from synthetic but not needed
    else:
        x, N = load_binary_from_csv(
            args.csv,
            timestamp_col=args.timestamp_col,
            value_col=args.value_col,
            value_threshold=args.value_threshold,
            bin_minutes=args.bin_minutes,
            start_hour=args.start_hour,
            days_span=args.days_span,
        )
        if x.size == 0:
            raise SystemExit("No events parsed from CSV. Check columns or thresholds.")
        empirical = empirical_per_bin_mean(x, N)

    if N <= 0:
        raise SystemExit("N must be positive.")
    if args.l <= 0 or args.l > N:
        raise SystemExit("--l must be in [1, N].")

    prior = ModelPrior(N=N, l=args.l, gamma=1.0, pois_lambda=1.0)

    # -------------------- Fit --------------------
    if args.pt:
        samples_tau, cp_hist, mode_tau, log_posts = _fit_pt(
            x=x, prior=prior, iters=args.iters, burn=args.burn, thin=args.thin,
            seed=args.seed, T_hot=args.T_hot, swap_every=args.swap_every
        )
        # For saving NPZ, we reuse RJConfig as metadata container (OK even for PT).
        cfg_for_save = RJConfig(iters=args.iters, burn=args.burn, thin=args.thin, seed=args.seed)
    else:
        samples_tau, cp_hist, mode_tau, log_posts = _fit_rjmcmc(
            x=x, prior=prior, iters=args.iters, burn=args.burn, thin=args.thin, seed=args.seed
        )
        cfg_for_save = RJConfig(iters=args.iters, burn=args.burn, thin=args.thin, seed=args.seed)

    # Save NPZ (result + prior + cfg)
    args.outdir.mkdir(parents=True, exist_ok=True)
    npz_path = args.outdir / args.save_npz
    save_result_npz(
        npz_path,
        samples_tau=samples_tau,
        log_posteriors=log_posts,
        changepoint_hist=cp_hist,
        mode_tau=mode_tau,
        prior_obj=prior,
        cfg_obj=cfg_for_save,
    )

    # Build a model instance for plotting (needs prior)
    model = WithinPeriodCPD(prior)

    # Plots
    _save_all_plots(
        prior=prior,
        samples_tau=samples_tau,
        cp_hist=cp_hist,
        mode_tau=mode_tau,
        model=model,
        outdir=args.outdir,
        start_hour=args.start_hour,
        hours_step=6,
        empirical_mean=empirical,
        title_suffix=" (demo)" if args.demo else "",
    )

    print(f"[OK] Saved NPZ: {npz_path}")
    print(f"[OK] Plots written to: {args.outdir}")


if __name__ == "__main__":
    main()
