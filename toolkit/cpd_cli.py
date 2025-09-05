# cpd_cli.py
# MIT License
"""
Universal CLI wrapper for Change-Point & State-Space Toolkit
Supports all methods with CSV I/O, plotting, and result export.

Examples:
    # E-Divisive multivariate CPD
    python cpd_cli.py edivisive --input data.csv --columns x,y,z --output results/

    # Kernel CPD with RBF
    python cpd_cli.py kcp --input data.csv --kernel rbf --output results/

    # RFF KCP for large datasets
    python cpd_cli.py rff-kcp --input data.csv --n-features 512 --output results/

    # HSMM with Gaussian emissions
    python cpd_cli.py hsmm --input data.csv --n-states 3 --emission gaussian --output results/

    # Within-period CPD for daily patterns
    python cpd_cli.py within-period --input activity.csv --bin-minutes 15 --output results/
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


def load_csv_data(
    filepath: str,
    columns: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    skip_rows: int = 0,
    max_rows: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], Optional[np.ndarray]]:
    """
    Load data from CSV with flexible column selection.

    Returns:
        data: (n_samples, n_features) array
        column_names: list of selected column names
        timestamps: optional timestamp array if timestamp_col provided
    """
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if columns:
            selected_cols = [c.strip() for c in columns.split(",")]
        else:
            selected_cols = [col for col in fieldnames if col != timestamp_col]

        # Validate columns exist
        missing = set(selected_cols) - set(fieldnames)
        if missing:
            raise ValueError(f"Columns not found in CSV: {missing}")

        if timestamp_col and timestamp_col not in fieldnames:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found")

        rows = []
        timestamps = []

        for i, row in enumerate(reader):
            if i < skip_rows:
                continue
            if max_rows and len(rows) >= max_rows:
                break

            try:
                data_row = [float(row[col]) for col in selected_cols]
                rows.append(data_row)

                if timestamp_col:
                    timestamps.append(row[timestamp_col])
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping row {i + 1}: {e}")
                continue

    data = np.array(rows)
    ts_array = np.array(timestamps) if timestamps else None

    return data, selected_cols, ts_array


def save_results(
    output_dir: Path,
    results: Dict[str, Any],
    plots: Dict[str, plt.Figure],
    method_name: str,
) -> None:
    """Save results, plots, and metadata to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save numerical results as NPZ
    np.savez_compressed(
        output_dir / f"{method_name}_results.npz",
        **{k: v for k, v in results.items() if isinstance(v, np.ndarray)},
    )

    # Save metadata as JSON
    metadata = {
        k: v for k, v in results.items() if not isinstance(v, np.ndarray) and not callable(v)
    }
    with open(output_dir / f"{method_name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Save plots
    for name, fig in plots.items():
        fig.savefig(output_dir / f"{method_name}_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Save CSV summary
    if "change_points" in results:
        cps = results["change_points"]
        if len(cps) > 0:
            with open(output_dir / f"{method_name}_changepoints.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["change_point_index", "position"])
                for i, cp in enumerate(cps):
                    writer.writerow([i, cp])




def run_edivisive(args) -> Tuple[Dict[str, Any], Dict[str, plt.Figure]]:
    """Run E-Divisive algorithm."""
    from changepoint_lab import edivisive
    from changepoint_lab.common.plotting.edivisive_plotting import (
        plot_segments_1d,
        plot_scree_edivisive,
    )

    data, columns, _ = load_csv_data(args.input, columns=args.columns)

    result = edivisive(
        data,
        alpha=args.alpha,
        min_size=args.min_size,
        R=args.R,
        significance=args.significance,
        resample=args.resample,
        block_size=args.block_size,
        seed=args.seed,
    )

    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_segments_1d(data[:, 0], result, ax=axes[0])
    plot_scree_edivisive(result, ax=axes[1])

    results = {
        "change_points": np.array(result.change_points),
        "method": "edivisive",
        "alpha": args.alpha,
        "significance": args.significance,
        "n_permutations": args.R,
    }

    return results, {"result": fig}


def run_kcp(args) -> Tuple[Dict[str, Any], Dict[str, plt.Figure]]:
    """Run Kernel Change-Point Detection."""
    from changepoint_lab.algorithms.kernel.kcp_core import (
        gram_rbf,
        gram_linear,
        build_kernel_prefix,
        kcp_penalized,
        kcp_select_bic,
        kcp_fixed_m,
    )
    from changepoint_lab.common.plotting import plot_segments_1d, plot_model_scree

    data, columns, _ = load_csv_data(args.input, columns=args.columns)

    # Build kernel matrix
    if args.kernel == "rbf":
        if args.bandwidth_cv:
            from changepoint_lab.algorithms.kernel.bandwidth_cv import select_rbf_bandwidth_cv

            gamma = select_rbf_bandwidth_cv(data, cv_folds=args.cv_folds)
            K, _ = gram_rbf(data, gamma=gamma)
        else:
            K, gamma = gram_rbf(data, gamma=args.gamma)
    elif args.kernel == "linear":
        K = gram_linear(data)
        gamma = args.penalty
    else:
        raise ValueError(f"Unknown kernel: {args.kernel}")

    prefix = build_kernel_prefix(K)

    if args.n_segments is None:
        # Penalized approach
        result = kcp_penalized(prefix, gamma=gamma, min_size=args.min_size, method=args.method)
    else:
        # Fixed number of segments
        result = kcp_fixed_m(prefix, m=args.n_segments, min_size=args.min_size)

    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    # First axis: data with detected change points (use first dimension)
    plot_segments_1d(data[:, 0], result.edges, ax=axes[0], title="Detected change points")
    # Second axis: cost vs segments if model selection available
    if hasattr(result, "model_sel"):
        plot_model_scree(result.model_sel, ax=axes[1])

    results = {
        "change_points": np.array(result.change_points),
        "cost": result.cost,
        "method": "kcp",
        "kernel": args.kernel,
        "penalty": gamma,
    }

    return results, {"result": fig}


def run_rff_kcp(args) -> Tuple[Dict[str, Any], Dict[str, plt.Figure]]:
    """Run RFF Kernel Change-Point Detection."""
    from changepoint_lab.algorithms.kernel.kcp_rff import (
        RFFConfig,
        rbf_rff_map,
        build_feature_prefix,
        rff_kcp_penalized,
        rff_kcp_fixed_m,
    )
    from changepoint_lab.algorithms.kernel.rff_variants import (
        OrthogonalRFFConfig,
        QuasiMCRFFConfig,
    )

    data, columns, _ = load_csv_data(args.input, columns=args.columns)

    # Select RFF variant
    if args.rff_type == "standard":
        rff_config = RFFConfig(n_features=args.n_features, seed=args.seed)
    elif args.rff_type == "orthogonal":
        rff_config = OrthogonalRFFConfig(n_features=args.n_features, seed=args.seed)
    elif args.rff_type == "quasi_mc":
        rff_config = QuasiMCRFFConfig(n_features=args.n_features, seed=args.seed)
    else:
        raise ValueError(f"Unknown RFF type: {args.rff_type}")

    # Build RFF mapping
    if args.bandwidth_cv:
        from changepoint_lab.algorithms.kernel.bandwidth_cv import select_rbf_bandwidth_cv

        sigma = select_rbf_bandwidth_cv(data, cv_folds=args.cv_folds)
        rff = rbf_rff_map(data, rff_config, sigma=sigma)
    else:
        rff = rbf_rff_map(data, rff_config, sigma=args.sigma)

    prefix = build_feature_prefix(rff.Z)

    if args.n_segments is None:
        result = rff_kcp_penalized(
            prefix, gamma_pen=args.penalty, min_size=args.min_size, method=args.method
        )
    else:
        result = rff_kcp_fixed_m(prefix, m=args.n_segments, min_size=args.min_size)

    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    # Plot original data with change points
    axes[0].plot(data)
    for cp in result.change_points:
        axes[0].axvline(cp, color="red", linestyle="--", alpha=0.7)
    axes[0].set_title("Data with Detected Change Points")
    axes[0].set_ylabel("Values")

    # Plot RFF features (sample)
    n_plot = min(10, rff.Z.shape[1])
    axes[1].plot(rff.Z[:, :n_plot])
    axes[1].set_title(f"RFF Features (showing {n_plot}/{rff.Z.shape[1]})")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Feature Values")

    results = {
        "change_points": np.array(result.change_points),
        "cost": result.cost,
        "method": "rff_kcp",
        "n_features": args.n_features,
        "rff_type": args.rff_type,
        "bandwidth": rff.gamma,
    }

    return results, {"result": fig}


def run_hsmm(args) -> Tuple[Dict[str, Any], Dict[str, plt.Figure]]:
    """Run Hidden Semi-Markov Model."""
    from changepoint_lab import HSMM, HSMMConfig, HSMMParams, PoissonDur
    from changepoint_lab.algorithms.state_space.emissions.gaussian_full import (
        GaussianFullEmissions,
    )  # Will implement below
    from changepoint_lab.algorithms.state_space.emissions.ar_emissions import (
        AREmissions,
    )  # Will implement below

    data, columns, _ = load_csv_data(args.input, columns=args.columns)

    # Build emission model
    if args.emission == "gaussian_diag":
        from gaussian_diag import estimate_by_kmeanspp, gaussian_diag_loglik

        em = estimate_by_kmeanspp(data, args.n_states, n_init=5)
        loglik = gaussian_diag_loglik(data, em)
    elif args.emission == "gaussian_full":
        em = GaussianFullEmissions(args.n_states)
        em.initialize_kmeans(data)
        loglik = em.compute_loglik(data)
    elif args.emission == "ar":
        em = AREmissions(args.n_states, order=args.ar_order)
        em.initialize(data)
        loglik = em.compute_loglik(data)
    else:
        raise ValueError(f"Unknown emission type: {args.emission}")

    # Initialize HSMM
    config = HSMMConfig(K=args.n_states, Dmax=args.max_duration, min_duration=args.min_duration)

    pi0 = np.full(args.n_states, 1.0 / args.n_states)
    A0 = np.full((args.n_states, args.n_states), 1.0 / (args.n_states - 1))
    np.fill_diagonal(A0, 0.0)

    duration_params = PoissonDur(lam=np.full(args.n_states, args.mean_duration))
    params = HSMMParams(pi=pi0, A=A0, duration=("poisson", duration_params))

    hsmm = HSMM(config, params)

    # Fit model
    fitted_params, ll_trace = hsmm.fit(loglik, max_iter=args.max_iter)

    # Decode states
    states, durations = hsmm.decode_viterbi(loglik)

    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot data
    axes[0].plot(data)
    axes[0].set_title("Input Data")
    axes[0].set_ylabel("Values")

    # Plot states
    axes[1].plot(states)
    axes[1].set_title("Decoded States")
    axes[1].set_ylabel("State")

    # Plot log-likelihood trace
    axes[2].plot(ll_trace)
    axes[2].set_title("Log-Likelihood During Training")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Log-Likelihood")

    results = {
        "states": states,
        "durations": durations,
        "log_likelihood_trace": np.array(ll_trace),
        "method": "hsmm",
        "n_states": args.n_states,
        "emission_type": args.emission,
        "final_loglik": ll_trace[-1] if ll_trace else 0,
    }

    return results, {"result": fig}


def run_within_period(args) -> Tuple[Dict[str, Any], Dict[str, plt.Figure]]:
    """Run Within-Period Change-Point Detection."""
    from within_period.within_period_cpd import WithinPeriodCPD, ModelPrior, RJConfig
    from changepoint_lab.common.io.data_loader import (
        load_binary_from_csv,
        empirical_per_bin_mean,
    )
    from changepoint_lab.common.plotting.plotting_helpers import (
        plot_changepoint_posterior_mass,
        plot_pointwise_bands,
    )

    # Load binary time series
    x, N = load_binary_from_csv(
        args.input,
        timestamp_col=args.timestamp_col,
        bin_minutes=args.bin_minutes,
        start_hour=args.start_hour,
    )

    empirical = empirical_per_bin_mean(x, N)

    # Set up model
    prior = ModelPrior(
        N=N, l=args.min_segment_length, gamma=args.gamma, pois_lambda=args.pois_lambda
    )
    model = WithinPeriodCPD(prior)

    # Configure sampler
    config = RJConfig(
        iters=args.iters,
        burn=args.burn,
        thin=args.thin,
        seed=args.seed,
    )

    # Fit model
    result = model.fit(x, config)

    # Generate posterior summary
    pw = model.pointwise_posterior_summary_from_samples(
        result.samples_tau,
        draws_per_sample=2,
        credible=0.95,
    )

    # Create plots
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    plot_changepoint_posterior_mass(
        cp_hist=result.changepoint_hist,
        num_samples=len(result.samples_tau),
        N=prior.N,
        tau_map=result.mode_tau,
        start_hour=args.start_hour,
        ax=ax1,
    )

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    plot_pointwise_bands(
        pw=pw,
        tau=result.mode_tau,
        start_hour=args.start_hour,
        ax=ax2,
    )
    ax2.plot(np.arange(N), empirical, linestyle=":", label="Empirical", alpha=0.8)
    ax2.legend()

    results = {
        "change_points": np.array(result.mode_tau),
        "samples_kept": len(result.samples_tau),
        "posterior_median": pw["median"],
        "posterior_lower": pw["lower"],
        "posterior_upper": pw["upper"],
        "empirical_mean": empirical,
        "method": "within_period_cpd",
        "bins_per_day": N,
    }

    return results, {"posterior_mass": fig1, "pointwise_bands": fig2}


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Change-Point & State-Space Toolkit CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Global arguments
    parser.add_argument(
        "--output", "-o", type=str, default="results", help="Output directory for results and plots"
    )
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")

    subparsers = parser.add_subparsers(dest="method", help="CPD method to run")
    subparsers.required = True


    # E-Divisive
    ed_parser = subparsers.add_parser("edivisive", help="E-Divisive multivariate CPD")
    ed_parser.add_argument(
        "--columns", type=str, required=True, help="Comma-separated column names"
    )
    ed_parser.add_argument("--alpha", type=float, default=1.0, help="Energy statistic parameter")
    ed_parser.add_argument("--min-size", type=int, default=30, help="Minimum segment size")
    ed_parser.add_argument("--R", type=int, default=499, help="Number of permutations")
    ed_parser.add_argument("--significance", type=float, default=0.05, help="Significance level")
    ed_parser.add_argument(
        "--resample",
        choices=["iid", "block-permutation", "circular-block-bootstrap"],
        default="circular-block-bootstrap",
        help="Resampling method",
    )
    ed_parser.add_argument(
        "--block-size", type=int, help="Block size for bootstrap (auto if not set)"
    )

    # Kernel CPD
    kcp_parser = subparsers.add_parser("kcp", help="Kernel Change-Point Detection")
    kcp_parser.add_argument(
        "--columns", type=str, required=True, help="Comma-separated column names"
    )
    kcp_parser.add_argument(
        "--kernel", choices=["rbf", "linear"], default="rbf", help="Kernel type"
    )
    kcp_parser.add_argument("--gamma", type=float, help="RBF kernel bandwidth (auto if not set)")
    kcp_parser.add_argument("--penalty", type=float, help="Penalty parameter (auto if not set)")
    kcp_parser.add_argument("--min-size", type=int, default=20, help="Minimum segment size")
    kcp_parser.add_argument(
        "--method", choices=["pelt", "op"], default="pelt", help="Optimization method"
    )
    kcp_parser.add_argument(
        "--n-segments", type=int, help="Fixed number of segments (overrides penalty)"
    )
    kcp_parser.add_argument(
        "--bandwidth-cv", action="store_true", help="Use cross-validation for bandwidth selection"
    )
    kcp_parser.add_argument(
        "--cv-folds", type=int, default=5, help="Number of CV folds for bandwidth selection"
    )

    # RFF KCP
    rff_parser = subparsers.add_parser("rff-kcp", help="Random Fourier Features KCP")
    rff_parser.add_argument(
        "--columns", type=str, required=True, help="Comma-separated column names"
    )
    rff_parser.add_argument("--n-features", type=int, default=512, help="Number of random features")
    rff_parser.add_argument(
        "--rff-type",
        choices=["standard", "orthogonal", "quasi_mc"],
        default="standard",
        help="RFF variant",
    )
    rff_parser.add_argument("--sigma", type=float, help="RBF bandwidth (auto if not set)")
    rff_parser.add_argument("--penalty", type=float, help="Penalty parameter (auto if not set)")
    rff_parser.add_argument("--min-size", type=int, default=20, help="Minimum segment size")
    rff_parser.add_argument(
        "--method", choices=["pelt", "op"], default="pelt", help="Optimization method"
    )
    rff_parser.add_argument(
        "--bandwidth-cv", action="store_true", help="Use cross-validation for bandwidth selection"
    )
    rff_parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds")

    # HSMM
    hsmm_parser = subparsers.add_parser("hsmm", help="Hidden Semi-Markov Model")
    hsmm_parser.add_argument(
        "--columns", type=str, required=True, help="Comma-separated column names"
    )
    hsmm_parser.add_argument("--n-states", type=int, required=True, help="Number of hidden states")
    hsmm_parser.add_argument(
        "--emission",
        choices=["gaussian_diag", "gaussian_full", "ar"],
        default="gaussian_diag",
        help="Emission distribution",
    )
    hsmm_parser.add_argument("--max-duration", type=int, default=100, help="Maximum state duration")
    hsmm_parser.add_argument("--min-duration", type=int, default=1, help="Minimum state duration")
    hsmm_parser.add_argument(
        "--mean-duration",
        type=float,
        default=20,
        help="Expected state duration (for Poisson prior)",
    )
    hsmm_parser.add_argument("--ar-order", type=int, default=1, help="AR order (for AR emissions)")
    hsmm_parser.add_argument("--max-iter", type=int, default=100, help="Maximum EM iterations")

    # Within-Period CPD
    wp_parser = subparsers.add_parser("within-period", help="Within-Period CPD for daily patterns")
    wp_parser.add_argument(
        "--timestamp-col", type=str, default="timestamp", help="Timestamp column name"
    )
    wp_parser.add_argument("--bin-minutes", type=int, default=15, help="Minutes per bin")
    wp_parser.add_argument("--start-hour", type=int, default=0, help="Hour that maps to bin 0")
    wp_parser.add_argument(
        "--min-segment-length", type=int, default=4, help="Minimum segment length in bins"
    )
    wp_parser.add_argument("--gamma", type=float, default=1.0, help="Dirichlet shape parameter")
    wp_parser.add_argument(
        "--pois-lambda", type=float, default=1.0, help="Poisson prior on number of segments"
    )
    wp_parser.add_argument("--iters", type=int, default=20000, help="MCMC iterations")
    wp_parser.add_argument("--burn", type=int, default=10000, help="Burn-in iterations")
    wp_parser.add_argument("--thin", type=int, default=10, help="Thinning interval")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    print(f"Running {args.method} on {args.input}")
    print(f"Results will be saved to {args.output}")

    # Route to appropriate method
    method_map = {
        "edivisive": run_edivisive,
        "kcp": run_kcp,
        "rff-kcp": run_rff_kcp,
        "hsmm": run_hsmm,
        "within-period": run_within_period,
    }

    try:
        results, plots = method_map[args.method](args)

        # Save all outputs
        output_dir = Path(args.output)
        save_results(output_dir, results, plots, args.method)

        print(f"\nResults saved to {output_dir}")
        if "change_points" in results:
            cps = results["change_points"]
            print(f"Found {len(cps)} change points: {list(cps)}")

    except Exception as e:
        print(f"Error running {args.method}: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
