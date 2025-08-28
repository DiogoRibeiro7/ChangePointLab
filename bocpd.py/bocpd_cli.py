# bocpd_cli.py
# MIT License
# (c) 2025

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from bocpd import BOCPD, BOCPDConfig, ConstantHazard, ScheduledHazard, BoostedBoundaryHazard
from bocpd_plotting import plot_run_length_heatmap, plot_cp_probability
from data_loader import load_binary_from_csv  # from your earlier module


def _build_synth(N: int, days: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Simple day pattern: low activity first quarter, high else; flip every ~N steps randomly
    p = np.full(N, 0.35, dtype=float)
    p[: N // 4] = 0.05
    X = np.concatenate([rng.binomial(1, p).astype(bool) for _ in range(days)])
    return X


def main() -> None:
    ap = argparse.ArgumentParser(description="BOCPD (Beta–Bernoulli) streaming CLI")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Synthetic demo.")
    mode.add_argument("--csv", type=str, help="Path to CSV of timestamps (events).")

    # Binning
    ap.add_argument("--bin-minutes", type=int, default=15, help="Minutes per bin (CSV).")
    ap.add_argument("--start-hour", type=int, default=0, help="Hour mapped to index 0 (CSV).")
    ap.add_argument("--timestamp-col", type=str, default="timestamp", help="CSV timestamp column.")
    ap.add_argument("--value-col", type=str, default=None, help="Optional numeric column to threshold for events.")
    ap.add_argument("--value-threshold", type=float, default=0.0, help="Threshold for value_col.")
    ap.add_argument("--days-span", type=int, default=None, help="Force exact #days window from min date (CSV).")

    # Prior & truncation
    ap.add_argument("--alpha0", type=float, default=1.0, help="Beta prior alpha0.")
    ap.add_argument("--beta0", type=float, default=1.0, help="Beta prior beta0.")
    ap.add_argument("--Rmax", type=int, default=512, help="Max run-length support.")

    # Hazard
    ap.add_argument("--mean-rl", type=float, default=96.0, help="Mean run length for ConstantHazard.")
    ap.add_argument("--schedule", type=str, default=None,
                    help="Optional comma-separated hazard schedule per period (overrides mean-rl).")
    ap.add_argument("--period", type=int, default=None, help="Period for schedule / boundary boost.")
    ap.add_argument("--boost-boundary", type=str, default=None,
                    help="Optional comma-separated boundary indices to boost (e.g., '0' for t%%N==0).")
    ap.add_argument("--boost-factor", type=float, default=10.0, help="Boundary hazard multiplier if boosting.")

    # Output
    ap.add_argument("--outdir", type=Path, default=Path("out_bocpd"), help="Directory for outputs.")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed (demo).")
    ap.add_argument("--days", type=int, default=30, help="Days (demo).")
    ap.add_argument("--cp-threshold", type=float, default=0.6, help="Threshold to flag CP events.")

    args = ap.parse_args()

    # ---- Build data ----
    if args.demo:
        N = int(args.period) if args.period else 96
        x = _build_synth(N=N, days=args.days, seed=args.seed)
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
            raise SystemExit("No events parsed from CSV. Check inputs.")

    # ---- Hazard selection ----
    if args.schedule is not None:
        values = [float(v.strip()) for v in args.schedule.split(",") if v.strip()]
        if args.period is None or int(args.period) != len(values):
            raise SystemExit("--period must equal len(--schedule).")
        base = ScheduledHazard(values, period=int(args.period))
    else:
        base = ConstantHazard(mean_run_length=float(args.mean_rl))

    if args.boost_boundary is not None:
        if args.period is None:
            raise SystemExit("--period required for --boost-boundary.")
        idx = frozenset(int(v.strip()) for v in args.boost_boundary.split(",") if v.strip())
        hazard = BoostedBoundaryHazard(base=base, period=int(args.period), boundary_indices=idx,
                                       boost_factor=float(args.boost_factor))
    else:
        hazard = base

    # ---- Run BOCPD ----
    cfg = BOCPDConfig(alpha0=float(args.alpha0), beta0=float(args.beta0),
                      max_run_length=int(args.Rmax), store_run_length_posterior=True)
    model = BOCPD(hazard, cfg)
    res = model.run(x)

    # ---- Save plots ----
    args.outdir.mkdir(parents=True, exist_ok=True)
    if res.run_length_posterior is not None:
        plot_run_length_heatmap(res.run_length_posterior, title="Run-length posterior (BOCPD)")
        plt.savefig(args.outdir / "rl_posterior.png", bbox_inches="tight", dpi=150)
        plt.close()

    plot_cp_probability(res.cp_prob, title="P(r_t=0 | x_{1:t})")
    plt.axhline(args.cp_threshold, linestyle="--")
    plt.savefig(args.outdir / "cp_probability.png", bbox_inches="tight", dpi=150)
    plt.close()

    # ---- Simple event report ----
    cps = np.nonzero(res.cp_prob >= args.cp_threshold)[0].tolist()
    (args.outdir / "summary.txt").write_text(
        f"N={N}\n"
        f"alpha0={args.alpha0}, beta0={args.beta0}\n"
        f"Rmax={args.Rmax}\n"
        f"CP threshold={args.cp_threshold}\n"
        f"#CP flagged={len(cps)}\n"
        f"indices={cps}\n"
    )
    print(f"[OK] Wrote heatmap + CP plot + summary to: {args.outdir}")
