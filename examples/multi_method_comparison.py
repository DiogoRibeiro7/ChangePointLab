"""Comprehensive examples comparing changepoint methods on synthetic data.

This script generates multiple synthetic datasets exhibiting different types of
changepoints (abrupt, gradual, variance, periodic, and compositional). For each
scenario the script runs a selection of detectors from the toolkit – BOCPD,
PELT, E-Divisive, HSMM/SD-HMM, and Within-Period CPD – using varied
configurations to highlight their strengths and differences.  Results are
plotted side by side and simple performance metrics (runtime and changepoint
error) are reported.

The companion notebook in ``docs/notebooks/multi_method_comparison.ipynb``
uses the helper functions defined here to create interactive figures.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Unified API imports
from changepoint_toolkit import (
    BOCPD,
    BOCPDConfig,
    ConstantHazard,
    BoostedBoundaryHazard,
    PELT,
    pelt,
    NormalMeanKnownVar,
    NormalMeanVarUnknown,
    edivisive,
    HSMM,
    HSMMConfig,
    SDHMM,
    SDHMMConfig,
    WithinPeriodCPD,
    ModelPrior,
    RJConfig,
)

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def abrupt_mean_shift(n: int = 200) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Mean shift in Gaussian data."""
    rng = np.random.default_rng(0)
    data = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(3, 1, n // 2)])
    binary = (data > 1.5).astype(int)
    return data, binary, [n // 2]

def variance_shift(n: int = 200) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Variance shift in Gaussian data."""
    rng = np.random.default_rng(1)
    data = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(0, 3, n // 2)])
    binary = (data > 0).astype(int)
    return data, binary, [n // 2]

def gradual_mean_change(n: int = 200) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Linear trend change."""
    rng = np.random.default_rng(2)
    t = np.arange(n)
    data = np.piecewise(
        t,
        [t < n // 2, t >= n // 2],
        [lambda x: rng.normal(0.0 + 0.02 * x, 1.0), lambda x: rng.normal(2.0 + 0.02 * x, 1.0)],
    )
    binary = (data > 1.0).astype(int)
    return data, binary, [n // 2]

def periodic_binary(n_periods: int = 10, period: int = 24) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Periodic binary sequence with a change in active hours."""
    rng = np.random.default_rng(3)
    data = np.zeros(n_periods * period, dtype=int)
    for d in range(n_periods):
        start = d * period
        active = slice(start + 8, start + 12)
        data[active] = rng.binomial(1, 0.8, active.stop - active.start)
    # Introduce different behaviour halfway
    mid = (n_periods // 2) * period
    data[mid + 12 : mid + 16] = rng.binomial(1, 0.8, 4)
    return data.astype(float), data, [mid]

def compositional_example(n: int = 200) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Compositional data with a change in Dirichlet parameters."""
    rng = np.random.default_rng(4)
    comp1 = rng.dirichlet([10, 5], n // 2)
    comp2 = rng.dirichlet([5, 10], n // 2)
    data = np.vstack([comp1, comp2])
    # For BOCPD/PELT, derive a univariate proxy
    cont = data[:, 0]
    binary = (cont > 0.5).astype(int)
    return cont, binary, [n // 2]

# ---------------------------------------------------------------------------
# Helper routines
# ---------------------------------------------------------------------------

def _runtime(fn: Callable[[], Tuple[List[int], float]]) -> Tuple[List[int], float]:
    start = time.perf_counter()
    cps = fn()
    end = time.perf_counter()
    return cps, end - start

def _plot_series(ax, series: np.ndarray, cps: List[int], label: str) -> None:
    ax.plot(series, label=label)
    for cp in cps:
        ax.axvline(cp, color="red", linestyle="--", alpha=0.5)
    ax.legend()

# ---------------------------------------------------------------------------
# Detector wrappers
# ---------------------------------------------------------------------------

def run_bocpd(binary: np.ndarray) -> List[int]:
    model = BOCPD(ConstantHazard(mean_run_length=50), BOCPDConfig(max_run_length=200))
    res = model.run(binary)
    return np.where(res.cp_prob > 0.5)[0].tolist()

def run_bocpd_boosted(binary: np.ndarray) -> List[int]:
    base = ConstantHazard(mean_run_length=50)
    hazard = BoostedBoundaryHazard(base, period=50, boundary_indices={0}, boost_factor=5.0)
    model = BOCPD(hazard, BOCPDConfig(max_run_length=200))
    res = model.run(binary)
    return np.where(res.cp_prob > 0.5)[0].tolist()

def run_pelt_known(data: np.ndarray) -> List[int]:
    cost = NormalMeanKnownVar(sigma2=1.0)
    res = pelt(data, cost, penalty=1.0)
    return res.change_points

def run_pelt_unknown(data: np.ndarray) -> List[int]:
    cost = NormalMeanVarUnknown()
    res = pelt(data, cost, penalty=1.0)
    return res.change_points

def run_edivisive_default(data: np.ndarray) -> List[int]:
    res = edivisive(data, R=10, alpha=1.0)
    return res.change_points.tolist()

def run_edivisive_strict(data: np.ndarray) -> List[int]:
    res = edivisive(data, R=100, alpha=1.5)
    return res.change_points.tolist()

def run_hsmm(data: np.ndarray) -> List[int]:
    """Simple HSMM with two Gaussian states and Poisson durations."""
    T = data.shape[0]
    loglik = np.vstack([
        -0.5 * (data - 0.0) ** 2,
        -0.5 * (data - 3.0) ** 2,
    ]).T
    params = HSMM.gaussian_example_params(K=2, Dmax=20)
    model = HSMM(HSMMConfig(K=2, Dmax=20, max_em_iters=5), params)
    z, _ = model.decode_viterbi(loglik)
    cps = np.where(np.diff(z) != 0)[0] + 1
    return cps.tolist()

def run_sdhmm(comp: np.ndarray) -> List[int]:
    model = SDHMM(SDHMMConfig(K=2, max_em_iters=5))
    z = model.fit_predict(comp)
    cps = np.where(np.diff(z) != 0)[0] + 1
    return cps.tolist()

def run_within_period(binary: np.ndarray, period: int) -> List[int]:
    model = WithinPeriodCPD(ModelPrior(N=period, l=3))
    res = model.fit(binary.astype(bool), RJConfig(iters=200, burn=50))
    return res.mode_tau

# ---------------------------------------------------------------------------
# Main comparison routine
# ---------------------------------------------------------------------------

def compare_all() -> None:
    scenarios = {
        "abrupt_mean": abrupt_mean_shift,
        "variance_shift": variance_shift,
        "gradual_mean": gradual_mean_change,
        "periodic_binary": periodic_binary,
        "compositional": compositional_example,
    }
    for name, gen in scenarios.items():
        data, binary, true_cps = gen()
        fig, ax = plt.subplots(3, 2, figsize=(12, 8))
        ax = ax.ravel()

        # BOCPD variants
        cps, t = _runtime(lambda: run_bocpd(binary))
        _plot_series(ax[0], binary, cps, f"BOCPD ({t:.3f}s)")
        cps, t = _runtime(lambda: run_bocpd_boosted(binary))
        _plot_series(ax[1], binary, cps, f"BOCPD boosted ({t:.3f}s)")

        # PELT cost variants
        cps, t = _runtime(lambda: run_pelt_known(data))
        _plot_series(ax[2], data, cps, f"PELT known-var ({t:.3f}s)")
        cps, t = _runtime(lambda: run_pelt_unknown(data))
        _plot_series(ax[3], data, cps, f"PELT unknown-var ({t:.3f}s)")

        # E-Divisive
        cps, t = _runtime(lambda: run_edivisive_default(data))
        _plot_series(ax[4], data, cps, f"E-Divisive R10 ({t:.3f}s)")
        cps, t = _runtime(lambda: run_edivisive_strict(data))
        _plot_series(ax[5], data, cps, f"E-Divisive R100 ({t:.3f}s)")

        fig.suptitle(f"Scenario: {name}; true CPs {true_cps}")
        plt.tight_layout()
        plt.savefig(f"{name}_comparison.png")
        plt.close(fig)

        # Additional detectors for specific scenarios
        if name != "periodic_binary":
            cps = run_hsmm(data)
            print(f"HSMM changepoints ({name}):", cps)
        if name == "compositional":
            cps = run_sdhmm(np.vstack([data, 1 - data]).T)
            print("SD-HMM changepoints:", cps)
        if name == "periodic_binary":
            cps = run_within_period(binary, period=24)
            print("Within-period changepoints:", cps)

if __name__ == "__main__":
    compare_all()
