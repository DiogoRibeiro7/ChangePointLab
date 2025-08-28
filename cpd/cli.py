# cli.py
# MIT License

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from within_period_cpd import WithinPeriodCPD, ModelPrior, RJConfig, Tau
from plotting_helpers import (
    plot_changepoint_posterior_mass,
    plot_pointwise_bands,
    plot_posterior_num_segments,
)
from diagnostics import posterior_num_segments


def run_demo(N: int, l: int, days: int, seed: int, iters: int, burn: int, thin: int, outdir: Path) -> None:
    rng = np.random.default_rng(seed)
    # True 2-seg pattern
    tau_true: Tau = (N // 4, (N - 2) % N)  # arbitrary demo
    p_lo, p_hi = 0.05, 0.35

    # Build phi over N
    phi = np.full(N, p_hi, dtype=float)
    a = (tau_true[-1] + 1) % N
    length = (tau_true[0] - tau_true[-1]) % N or N
    for k in range(length):
        phi[(a + k) % N] = p_lo

    # Generate data
    X = np.concatenate([rng.binomial(1, phi).astype(bool) for _ in range(days)])

    prior = ModelPrior(N=N, l=l, gamma=1.0, pois_lambda=1.0)
    model = WithinPeriodCPD(prior)
    cfg = RJConfig(iters=iters, burn=burn, thin=thin, seed=seed)

    result = model.fit(X, cfg)
    pw = model.pointwise_posterior_summary_from_samples(result.samples_tau, draws_per_sample=2, credible=0.95)

    outdir.mkdir(parents=True, exist_ok=True)

    # Plot CP mass
    ax1 = plot_changepoint_posterior_mass(
        cp_hist=result.changepoint_hist,
        num_samples=len(result.samples_tau),
        N=prior.N,
        tau_map=result.mode_tau,
        start_hour=0,
        hours_step=6,
        title="Changepoint posterior mass",
    )
    (outdir / "cp_mass.png").write_bytes(plt.gcf().canvas.tostring_rgb())  # ensure canvas initialized
    plt.savefig(outdir / "cp_mass.png", bbox_inches="tight", dpi=150)
    plt.close()

    # Plot pointwise bands
    ax2 = plot_pointwise_bands(
        pw=pw,
        tau=result.mode_tau,
        start_hour=0,
        hours_step=6,
        title="Pointwise posterior bands",
    )
    plt.savefig(outdir / "pointwise_bands.png", bbox_inches="tight", dpi=150)
    plt.close()

    # Posterior over m
    pm = posterior_num_segments(result.samples_tau)
    plot_posterior_num_segments(pm.m_values, pm.probs, title="Posterior over m")
    plt.savefig(outdir / "posterior_m.png", bbox_inches="tight", dpi=150)
    plt.close()

    # Stats
    (outdir / "summary.txt").write_text(
        f"MAP tau: {result.mode_tau}\n"
        f"kept samples: {len(result.samples_tau)}\n"
        f"posterior m: {list(zip(pm.m_values.tolist(), pm.probs.round(3).tolist()))}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Within-period CPD demo runner")
    ap.add_argument("--N", type=int, default=96, help="bins per day (default: 96 for 15-min)")
    ap.add_argument("--l", type=int, default=4, help="min segment length in bins")
    ap.add_argument("--days", type=int, default=30, help="number of synthetic days")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed")
    ap.add_argument("--iters", type=int, default=20000, help="MCMC iterations")
    ap.add_argument("--burn", type=int, default=10000, help="MCMC burn-in")
    ap.add_argument("--thin", type=int, default=10, help="MCMC thinning")
    ap.add_argument("--outdir", type=Path, default=Path("out"), help="output directory")
    args = ap.parse_args()
    run_demo(args.N, args.l, args.days, args.seed, args.iters, args.burn, args.thin, args.outdir)


if __name__ == "__main__":
    main()
