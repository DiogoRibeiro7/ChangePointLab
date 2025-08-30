"""Generate figures for CPDToolkit paper."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from bocpd.bocpd import BOCPD, ConstantHazard, BOCPDConfig
from edivisive.edivisive import edivisive
from hsmm.gaussian_diag import estimate_by_kmeanspp, gaussian_diag_loglik
from hsmm.hsmm import HSMM, HSMMConfig, HSMMParams, PoissonDur
from pelt.pelt import BetaBinomialCost, pelt, bic_penalty
from sdhmm.sdhmm import SDHMM, SDHMMConfig
from within_period.within_period_cpd import ModelPrior, RJConfig, WithinPeriodCPD

FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def synthetic_binary() -> tuple[np.ndarray, list[int]]:
    """Generate a binary sequence with two change points."""
    rng = np.random.default_rng(0)
    segments = [
        rng.binomial(1, 0.1, 200),
        rng.binomial(1, 0.9, 200),
        rng.binomial(1, 0.3, 200),
    ]
    data = np.concatenate(segments)
    return data, [200, 400]


def run_algorithms(x: np.ndarray) -> tuple[dict[str, list[int]], np.ndarray]:
    """Run all CPD algorithms on sequence ``x``."""
    results: dict[str, list[int]] = {}

    # PELT
    cost = BetaBinomialCost(alpha=1.0, beta=1.0)
    res_pelt = pelt(x, cost, penalty=bic_penalty(1, len(x)), min_seg_len=5)
    results["PELT"] = res_pelt.change_points

    # E-Divisive
    res_ediv = edivisive(x.astype(float), min_size=30, seed=0)
    results["E-Divisive"] = res_ediv.change_points.tolist()

    # BOCPD
    model = BOCPD(ConstantHazard(100), BOCPDConfig(max_run_length=200))
    res_bocpd = model.run(x)
    cp_prob = res_bocpd.cp_prob
    cps_bocpd = [
        t
        for t in range(1, len(cp_prob) - 1)
        if cp_prob[t] > 0.012 and cp_prob[t] > cp_prob[t - 1] and cp_prob[t] > cp_prob[t + 1]
    ]
    results["BOCPD"] = cps_bocpd

    # HSMM
    X1 = x[:, None]
    K = 3
    em_params = estimate_by_kmeanspp(X1, K, n_init=3, max_iter=50, allow_nan=False)
    L = gaussian_diag_loglik(X1, em_params)
    pi0 = np.full(K, 1.0 / K)
    A0 = np.full((K, K), 1.0 / (K - 1))
    np.fill_diagonal(A0, 0.0)
    dur = ("poisson", PoissonDur(lam=np.full(K, 100.0)))
    hsmm_model = HSMM(
        HSMMConfig(K=K, Dmax=200, min_duration=20, max_em_iters=20),
        HSMMParams(pi=pi0, A=A0, duration=dur),
    )
    params_fit, _ = hsmm_model.fit(L)
    z, _ = hsmm_model.decode_viterbi(L)
    cps_hsmm = np.where(np.diff(z) != 0)[0] + 1
    results["HSMM"] = cps_hsmm.tolist()

    # SD-HMM
    X2 = np.column_stack([x, 1 - x]).astype(float)
    X2 += 1e-3
    X2 /= X2.sum(axis=1, keepdims=True)
    sdhmm_model = SDHMM(SDHMMConfig(K=3, max_iter=30, min_iter=5, tol=1e-4))
    sdhmm_model.fit(X2)
    z_hat = sdhmm_model.viterbi(X2)
    cps_sdhmm = np.where(np.diff(z_hat) != 0)[0] + 1
    results["SD-HMM"] = cps_sdhmm.tolist()

    # Within-period CPD
    prior = ModelPrior(N=100, l=10)
    wp_model = WithinPeriodCPD(prior)
    wp_res = wp_model.fit(x, RJConfig(iters=200, burn=50, thin=5, seed=0))
    results["Within-period"] = list(wp_res.mode_tau)

    return results, cp_prob


def plot_comparison(x: np.ndarray, true_cps: list[int], results: dict[str, list[int]], cp_prob: np.ndarray) -> None:
    methods = [
        ("Data", true_cps),
        ("PELT", results["PELT"]),
        ("E-Divisive", results["E-Divisive"]),
        ("BOCPD", results["BOCPD"]),
        ("HSMM", results["HSMM"]),
        ("SD-HMM", results["SD-HMM"]),
        ("Within-period", results["Within-period"]),
    ]
    fig, axes = plt.subplots(len(methods), 1, figsize=(10, 12), sharex=True)
    for ax, (name, cps) in zip(axes, methods):
        ax.plot(x, color="0.3")
        for cp in cps:
            ax.axvline(cp, color="red", linestyle="--")
        ax.set_ylabel(name)
    axes[0].set_title("Synthetic binary sequence with detected change points")
    axes[-1].set_xlabel("Time index")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison.png", dpi=300)

    # Performance comparison: difference in number of detected CPs
    diff = {m: abs(len(cps) - len(true_cps)) for m, cps in results.items()}
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(diff.keys(), diff.values(), color="steelblue")
    ax2.set_ylabel("|#CP - true|")
    ax2.set_title("Detection count error by method")
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "performance.png", dpi=300)


def plot_method_flowchart() -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    def box(x, y, text):
        w, h = 0.15, 0.15
        rect = plt.Rectangle((x - w / 2, y - h / 2), w, h, fc="lightgray", ec="black")
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center")
        return (x, y)

    root = box(0.1, 0.5, "Online\nBOCPD")
    offline = box(0.4, 0.5, "Offline")
    pelt = box(0.6, 0.8, "PELT")
    ediv = box(0.6, 0.5, "E-Divisive")
    hsmm = box(0.6, 0.2, "HMM/HSMM")
    sdhmm = box(0.8, 0.2, "SD-HMM")
    within = box(0.8, 0.8, "Within-period")

    def arrow(a, b):
        ax.annotate("", b, a, arrowprops=dict(arrowstyle="->"))

    arrow((0.2, 0.5), offline)
    arrow(offline, pelt)
    arrow(offline, ediv)
    arrow(offline, hsmm)
    arrow(hsmm, sdhmm)
    arrow(pelt, within)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "methods_flowchart.png", dpi=300)


def plot_decision_tree() -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    def box(x, y, text):
        w, h = 0.2, 0.1
        rect = plt.Rectangle((x - w / 2, y - h / 2), w, h, fc="white", ec="black")
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center")
        return (x, y)

    start = box(0.5, 0.9, "Start")
    online = box(0.5, 0.75, "Streaming data?")
    bocpd = box(0.25, 0.6, "BOCPD")
    offline = box(0.75, 0.6, "Offline analysis")
    binary = box(0.75, 0.45, "Binary/\ncompositional?")
    pelt = box(0.6, 0.3, "PELT")
    ediv = box(0.9, 0.3, "E-Divisive")
    periodic = box(0.4, 0.45, "Periodic pattern?")
    within = box(0.25, 0.3, "Within-period")
    state = box(0.75, 0.15, "State-space model?")
    hsmm = box(0.6, 0.05, "HMM/HSMM")
    sdhmm = box(0.9, 0.05, "SD-HMM")

    def arrow(a, b, text=""):
        ax.annotate(text, b, a, arrowprops=dict(arrowstyle="->"),
                    ha="center", va="center")

    arrow(start, online)
    arrow(online, bocpd, "Yes")
    arrow(online, offline, "No")
    arrow(offline, binary, "Yes")
    arrow(offline, state, "No")
    arrow(binary, pelt, "No")
    arrow(binary, ediv, "Yes")
    arrow(state, hsmm, "Gaussian")
    arrow(state, sdhmm, "Compositional")
    arrow(bocpd, periodic)
    arrow(periodic, within, "Yes")
    arrow(periodic, pelt, "No")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "decision_tree.png", dpi=300)


def main() -> None:
    x, true_cps = synthetic_binary()
    results, cp_prob = run_algorithms(x)
    plot_comparison(x, true_cps, results, cp_prob)
    plot_method_flowchart()
    plot_decision_tree()


if __name__ == "__main__":
    main()
