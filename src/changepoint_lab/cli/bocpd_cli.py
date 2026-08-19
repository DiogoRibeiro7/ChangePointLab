# bocpd_cli.py
# MIT License
# (c) 2025

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from changepoint_lab._optional import require_matplotlib_pyplot
from changepoint_lab.algorithms.bayesian.bocpd import (
    BOCPD,
    BOCPDAlertConfig,
    BOCPDConfig,
    BoostedBoundaryHazard,
    ConstantHazard,
    Hazard,
    ScheduledHazard,
    extract_changepoint_alerts,
)
from changepoint_lab.common.io.data_loader import load_binary_from_csv


def _parse_schedule(s: Optional[str]) -> Optional[np.ndarray]:
    if not s:
        return None
    vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    if np.any((arr <= 0) | (arr >= 1)):
        raise ValueError("Schedule entries must be in (0,1)")
    return arr


def _parse_indices(s: Optional[str]) -> Optional[np.ndarray]:
    if not s:
        return None
    vals = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=int)
    if np.any(arr < 0):
        raise ValueError("Boundary indices must be >= 0")
    return arr


def _build_synth(N: int, days: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = np.array([0.05, 0.2, 0.05, 0.2], dtype=float)
    X = np.concatenate([rng.binomial(1, p).astype(bool) for _ in range(days)])
    return X


def main() -> None:
    ap = argparse.ArgumentParser(description="BOCPD (Beta–Bernoulli) streaming CLI")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Synthetic demo.")
    mode.add_argument("--csv", type=str, help="Path to CSV of timestamps (events).")

    # Binning
    ap.add_argument("--bin-minutes", type=int, default=15, help="Minutes per bin (CSV).")
    ap.add_argument("--start-hour", type=int, default=0, help="Local hour mapped to index 0 (CSV).")
    ap.add_argument("--timestamp-col", type=str, default="timestamp", help="CSV timestamp column.")
    ap.add_argument("--value-col", type=str, default=None, help="Optional numeric column to threshold for events.")
    ap.add_argument("--value-threshold", type=float, default=0.0, help="Threshold for value_col.")
    ap.add_argument("--days-span", type=int, default=None, help="Force exact #local days from first anchor (CSV).")
    ap.add_argument("--timezone", type=str, default=None, help="IANA timezone (e.g., Europe/Lisbon).")

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

    # Numerical robustness (matches your earlier config)
    ap.add_argument("--prune-eps", type=float, default=1e-6,
                    help="Tail-pruning threshold; relative to max unless --abs-prune.")
    ap.add_argument("--abs-prune", action="store_true",
                    help="Use absolute pruning (R[r] < eps -> 0) instead of relative.")

    # Output
    ap.add_argument("--outdir", type=Path, default=Path("out_bocpd"), help="Directory for outputs.")
    ap.add_argument("--out-csv", type=str, default=None, help="Optional path to write per-step results CSV.")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed (demo).")
    ap.add_argument("--days", type=int, default=30, help="Days (demo).")
    ap.add_argument("--cp-threshold", type=float, default=0.6, help="Threshold to flag CP events.")

    args = ap.parse_args()

    # ---- Build data ----
    bin_edges = None
    if args.demo:
        N = int(args.period) if args.period else 96
        x = _build_synth(N=N, days=args.days, seed=args.seed)
    else:
        x, N, bin_edges = load_binary_from_csv(
            args.csv,
            timestamp_col=args.timestamp_col,
            value_col=args.value_col,
            value_threshold=args.value_threshold,
            bin_minutes=args.bin_minutes,
            start_hour=args.start_hour,
            days_span=args.days_span,
            timezone=args.timezone,
            return_time_bins=True,
        )

    # ---- Hazard ----
    schedule = _parse_schedule(args.schedule)
    boost_idx = _parse_indices(args.boost_boundary)

    hazard: Hazard
    if schedule is not None:
        if args.period is None:
            raise ValueError("--schedule provided but --period is None")
        hazard = ScheduledHazard(schedule=schedule, period=int(args.period))
    else:
        hazard = ConstantHazard(mean_run_length=float(args.mean_rl))

    if boost_idx is not None:
        if args.period is None:
            raise ValueError("--boost-boundary provided but --period is None")
        hazard = BoostedBoundaryHazard(
            base=hazard,
            boundary_indices={int(i) for i in boost_idx},
            period=int(args.period),
            boost_factor=float(args.boost_factor),
        )

    # ---- Model ----
    cfg = BOCPDConfig(
        alpha0=float(args.alpha0),
        beta0=float(args.beta0),
        max_run_length=int(args.Rmax),
        store_run_length_posterior=True,
        prune_epsilon=float(args.prune_eps),
        prune_relative=(not args.abs_prune),
        alert_config=BOCPDAlertConfig(probability_threshold=float(args.cp_threshold)),
    )
    model = BOCPD(cfg=cfg, hazard=hazard)

    res = model.run(x.astype(bool))

    # ---- Save plots ----
    plt = require_matplotlib_pyplot("bocpd-cli plots", backend="Agg")
    from changepoint_lab.algorithms.bayesian.bocpd.plotting import (
        plot_cp_probability,
        plot_run_length_heatmap,
    )

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
    cps = extract_changepoint_alerts(res, cfg.alert_config).tolist()
    (args.outdir / "summary.txt").write_text(
        f"N={N}\n"
        f"alpha0={args.alpha0}, beta0={args.beta0}\n"
        f"Rmax={args.Rmax}\n"
        f"CP threshold={args.cp_threshold}\n"
        f"#CP flagged={len(cps)}\n"
        f"indices={cps}\n"
    )

    # ---- Optional per-step CSV (machine-readable) ----
    if args.out_csv:
        # Build timestamps (left edges) if we have bin edges from CSV mode
        timestamps = None
        if bin_edges is not None and len(bin_edges) >= 2:
            # Left edge for each bin
            timestamps = bin_edges[:-1].astype("datetime64[ns]")

        import csv
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "timestamp", "cp_prob", "map_run_length", "pred_mean", "is_cp"])
            for t in range(len(res.cp_prob)):
                ts_val = str(timestamps[t]) if timestamps is not None and t < len(timestamps) else ""
                is_cp = float(t in cps)
                w.writerow([t, ts_val, float(res.cp_prob[t]), int(res.map_run_length[t]), float(res.pred_mean[t]), is_cp])

    print(f"[OK] Wrote heatmap + CP plot + summary to: {args.outdir}")
    if args.out_csv:
        print(f"[OK] Wrote stepwise results to: {args.out_csv}")


if __name__ == "__main__":
    main()
