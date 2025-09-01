"""Scenario-based comparison of changepoint detectors.

Each function in this module generates a synthetic dataset and evaluates a
selection of detection methods on it.  The aim is to highlight strengths and
weaknesses of the different algorithms on common problems such as mean/variance
shifts, periodic patterns and compositional changes.

Plotting, metric computation and summary printing are provided by
``comparison_helpers`` so that these routines remain focused on the workflow for
each scenario.  The functions are intended to be short, documented examples that
can be referenced in the toolkit documentation and JOSS paper.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from changepoint_toolkit import (
    BOCPD,
    BOCPDConfig,
    BoostedBoundaryHazard,
    ConstantHazard,
    HSMM,
    HSMMConfig,
    ModelPrior,
    NormalMeanKnownVar,
    NormalMeanVarUnknown,
    RJConfig,
    SDHMM,
    SDHMMConfig,
    WithinPeriodCPD,
    edivisive,
    pelt,
)
from examples.comparison_helpers import compare_detectors, print_summary

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def abrupt_mean_shift(n: int = 200) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    rng = np.random.default_rng(0)
    data = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(3, 1, n // 2)])
    binary = (data > 1.5).astype(int)
    return data, binary, [n // 2]

def variance_shift(n: int = 200) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    rng = np.random.default_rng(1)
    data = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(0, 3, n // 2)])
    binary = (data > 0).astype(int)
    return data, binary, [n // 2]

def gradual_mean_change(n: int = 200) -> Tuple[np.ndarray, np.ndarray, List[int]]:
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
    rng = np.random.default_rng(3)
    data = np.zeros(n_periods * period, dtype=int)
    for d in range(n_periods):
        start = d * period
        active = slice(start + 8, start + 12)
        data[active] = rng.binomial(1, 0.8, active.stop - active.start)
    mid = (n_periods // 2) * period
    data[mid + 12 : mid + 16] = rng.binomial(1, 0.8, 4)
    return data.astype(float), data, [mid]

def compositional_example(n: int = 200) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    rng = np.random.default_rng(4)
    comp1 = rng.dirichlet([10, 5], n // 2)
    comp2 = rng.dirichlet([5, 10], n // 2)
    data = np.vstack([comp1, comp2])
    cont = data[:, 0]
    binary = (cont > 0.5).astype(int)
    return cont, binary, [n // 2]

# ---------------------------------------------------------------------------
# Detector wrappers
# ---------------------------------------------------------------------------

def run_bocpd(binary: np.ndarray) -> List[int]:
    model = BOCPD(ConstantHazard(mean_run_length=50), BOCPDConfig(max_run_length=200))
    res = model.run(binary)
    return np.where(res.cp_prob > 0.3)[0].tolist()

def run_bocpd_boosted(binary: np.ndarray) -> List[int]:
    base = ConstantHazard(mean_run_length=50)
    hazard = BoostedBoundaryHazard(base, period=50, boundary_indices={0}, boost_factor=5.0)
    model = BOCPD(hazard, BOCPDConfig(max_run_length=200))
    res = model.run(binary)
    return np.where(res.cp_prob > 0.3)[0].tolist()

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
    T = data.shape[0]
    loglik = np.vstack([
        -0.5 * (data - 0.0) ** 2,
        -0.5 * (data - 3.0) ** 2,
    ]).T
    params = HSMM.gaussian_example_params(K=2, Dmax=20)
    model = HSMM(HSMMConfig(K=2, Dmax=20, max_em_iters=5), params)
    z, _ = model.decode_viterbi(loglik)
    return (np.where(np.diff(z) != 0)[0] + 1).tolist()

def run_sdhmm(comp: np.ndarray) -> List[int]:
    model = SDHMM(SDHMMConfig(K=2, max_em_iters=5))
    z = model.fit_predict(comp)
    return (np.where(np.diff(z) != 0)[0] + 1).tolist()

def run_within_period(binary: np.ndarray, period: int) -> List[int]:
    model = WithinPeriodCPD(ModelPrior(N=period, l=3))
    res = model.fit(binary.astype(bool), RJConfig(iters=200, burn=50))
    return res.mode_tau

# ---------------------------------------------------------------------------
# Scenario functions
# ---------------------------------------------------------------------------

DISCUSSION: Dict[str, str] = {
    "abrupt_mean": "PELT and BOCPD handle abrupt mean shifts best as they assume piecewise constant means.",
    "variance_shift": "E-Divisive excels for variance changes thanks to its distribution-free energy distance.",
    "gradual_mean": "BOCPD with a boosted hazard can adapt to gradual drifts by encouraging short run lengths.",
    "periodic_binary": "Within-period CPD dominates because it respects the known periodic structure.",
    "compositional": "SD-HMM captures full compositional shifts better than univariate methods.",
}

def scenario_abrupt_mean() -> None:
    data, binary, true_cps = abrupt_mean_shift()
    detectors = [
        ("BOCPD const", lambda b=binary: run_bocpd(b), binary),
        ("BOCPD boosted", lambda b=binary: run_bocpd_boosted(b), binary),
        ("PELT known", lambda d=data: run_pelt_known(d), data),
        ("PELT unknown", lambda d=data: run_pelt_unknown(d), data),
        ("E-Div R10", lambda d=data: run_edivisive_default(d), data),
        ("E-Div R100", lambda d=data: run_edivisive_strict(d), data),
        ("HSMM", lambda d=data: run_hsmm(d), data),
    ]
    metrics = compare_detectors("abrupt_mean", detectors, true_cps)
    print_summary("abrupt_mean", metrics, DISCUSSION["abrupt_mean"])

def scenario_variance_shift() -> None:
    data, binary, true_cps = variance_shift()
    detectors = [
        ("BOCPD const", lambda b=binary: run_bocpd(b), binary),
        ("PELT known", lambda d=data: run_pelt_known(d), data),
        ("PELT unknown", lambda d=data: run_pelt_unknown(d), data),
        ("E-Div R10", lambda d=data: run_edivisive_default(d), data),
        ("E-Div R100", lambda d=data: run_edivisive_strict(d), data),
    ]
    metrics = compare_detectors("variance_shift", detectors, true_cps)
    print_summary("variance_shift", metrics, DISCUSSION["variance_shift"])

def scenario_gradual_mean() -> None:
    data, binary, true_cps = gradual_mean_change()
    detectors = [
        ("BOCPD const", lambda b=binary: run_bocpd(b), binary),
        ("BOCPD boosted", lambda b=binary: run_bocpd_boosted(b), binary),
        ("PELT known", lambda d=data: run_pelt_known(d), data),
        ("PELT unknown", lambda d=data: run_pelt_unknown(d), data),
    ]
    metrics = compare_detectors("gradual_mean", detectors, true_cps)
    print_summary("gradual_mean", metrics, DISCUSSION["gradual_mean"])

def scenario_periodic_binary() -> None:
    data, binary, true_cps = periodic_binary()
    detectors = [
        ("BOCPD const", lambda b=binary: run_bocpd(b), binary),
        ("BOCPD boosted", lambda b=binary: run_bocpd_boosted(b), binary),
        ("PELT known", lambda d=data: run_pelt_known(d), data),
        ("Within-period", lambda b=binary: run_within_period(b, period=24), binary),
    ]
    metrics = compare_detectors("periodic_binary", detectors, true_cps)
    print_summary("periodic_binary", metrics, DISCUSSION["periodic_binary"])

def scenario_compositional() -> None:
    data, binary, true_cps = compositional_example()
    comp = np.vstack([data, 1 - data]).T
    detectors = [
        ("BOCPD const", lambda b=binary: run_bocpd(b), binary),
        ("PELT known", lambda d=data: run_pelt_known(d), data),
        ("PELT unknown", lambda d=data: run_pelt_unknown(d), data),
        ("SD-HMM", lambda c=comp: run_sdhmm(c), data),
    ]
    metrics = compare_detectors("compositional", detectors, true_cps)
    print_summary("compositional", metrics, DISCUSSION["compositional"])


def main() -> None:
    scenario_abrupt_mean()
    scenario_variance_shift()
    scenario_gradual_mean()
    scenario_periodic_binary()
    scenario_compositional()


if __name__ == "__main__":
    main()
