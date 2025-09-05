import time
from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pytest

pytestmark = pytest.mark.slow

from changepoint_lab import edivisive, pelt
from changepoint_lab.algorithms.bayesian.bocpd import (
    BOCPD,
    BOCPDConfig,
    BoostedBoundaryHazard,
    ConstantHazard,
)
from changepoint_lab.algorithms.optimization.cost_functions import (
    BetaBinomialCost,
    NormalMeanVarUnknown,
)
from changepoint_lab.algorithms.state_space.hsmm import (
    HSMM,
    HSMMConfig,
    HSMMParams,
    PoissonDur,
)

from changepoint_lab.algorithms.state_space.emissions.gaussian_diag import (
    GaussianDiagParams,
    gaussian_diag_loglik,
)
from changepoint_lab.algorithms.state_space.emissions.gaussian_full import (
    GaussianFullParams,
    gaussian_full_loglik,
)
from changepoint_lab.algorithms.state_space.sdhmm import SDHMM, SDHMMConfig
from changepoint_lab.algorithms.bayesian.within_period import (
    ModelPrior,
    RJConfig,
    WithinPeriodCPD,
)

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _precision_recall_f1(true: Sequence[int], pred: Sequence[int], tol: int) -> tuple[float, float, float]:
    """Precision/recall/F1 with tolerance window."""
    true = list(true)
    pred = list(pred)
    matched: set[int] = set()
    tp = 0
    for cp in pred:
        for j, cp_t in enumerate(true):
            if j in matched:
                continue
            if abs(cp - cp_t) <= tol:
                tp += 1
                matched.add(j)
                break
    fp = len(pred) - tp
    fn = len(true) - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def _avg_delay(true: Sequence[int], pred: Sequence[int]) -> float:
    """Average detection delay for online methods."""
    delays = []
    pred = sorted(pred)
    for cp in true:
        future = [p for p in pred if p >= cp]
        if future:
            delays.append(future[0] - cp)
    return float(np.mean(delays)) if delays else float("inf")


def evaluate(
    method: Callable[[np.ndarray], Sequence[int]],
    data: np.ndarray,
    true_cps: Sequence[int],
    *,
    tol: int = 5,
    online: bool = False,
) -> dict[str, float]:
    start = time.perf_counter()
    pred = list(method(data))
    runtime = time.perf_counter() - start
    prec, rec, f1 = _precision_recall_f1(true_cps, pred, tol)
    delay = _avg_delay(true_cps, pred) if online else np.nan
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "delay": delay,
        "runtime": runtime,
        "cps": pred,
    }


def plot_performance(records: dict[str, dict[str, float]], metric: str = "f1"):
    """Visualize a chosen metric for each method."""
    fig, ax = plt.subplots()
    names = list(records)
    vals = [records[n][metric] for n in names]
    ax.bar(names, vals)
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{metric} comparison")
    return fig


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def binary_data(
    seg_lengths: Sequence[int], probs: Sequence[float], rng: np.random.Generator
) -> tuple[np.ndarray, list[int]]:
    data = []
    cps = []
    cum = 0
    for L, p in zip(seg_lengths, probs, strict=False):
        data.append(rng.binomial(1, p, size=L))
        cum += L
        cps.append(cum)
    cps.pop()  # last boundary is end of series
    return np.concatenate(data), cps


def continuous_data(
    means: Sequence[float], vars: Sequence[float], seg_lengths: Sequence[int], rng: np.random.Generator
) -> tuple[np.ndarray, list[int]]:
    data = []
    cps = []
    cum = 0
    for m, v, L in zip(means, vars, seg_lengths, strict=False):
        data.append(rng.normal(m, np.sqrt(v), size=L))
        cum += L
        cps.append(cum)
    cps.pop()
    return np.concatenate(data), cps


def periodic_data(
    period: int,
    n_cycles: int,
    cp1: int,
    cp2: int,
    break_cycle: int,
) -> tuple[np.ndarray, int, list[int]]:
    """Binary periodic sequence with structural break in within-period cp."""
    pattern1 = np.concatenate([np.zeros(cp1, dtype=int), np.ones(period - cp1, dtype=int)])
    pattern2 = np.concatenate([np.zeros(cp2, dtype=int), np.ones(period - cp2, dtype=int)])
    data = np.concatenate(
        [np.tile(pattern1, break_cycle), np.tile(pattern2, n_cycles - break_cycle)]
    )
    return data, period, sorted({cp1, cp2})


def compositional_data(
    alphas: Sequence[Sequence[float]], seg_lengths: Sequence[int], rng: np.random.Generator
) -> tuple[np.ndarray, list[int]]:
    data = []
    cps = []
    cum = 0
    for a, L in zip(alphas, seg_lengths, strict=False):
        data.append(rng.dirichlet(a, size=L))
        cum += L
        cps.append(cum)
    cps.pop()
    return np.vstack(data), cps


# ---------------------------------------------------------------------------
# Method wrappers
# ---------------------------------------------------------------------------

def run_bocpd_const(mean_rl: int) -> Callable[[np.ndarray], list[int]]:
    def _run(x: np.ndarray) -> list[int]:
        model = BOCPD(ConstantHazard(mean_rl), BOCPDConfig(max_run_length=500))
        res = model.run(x)
        positions = np.flatnonzero(res.cp_prob >= 0.05)
        out: list[int] = []
        for p in positions:
            if not out or p - out[-1] > 5:
                out.append(int(p))
        return out

    return _run


def run_bocpd_boost(period: int, boundary: int) -> Callable[[np.ndarray], list[int]]:
    def _run(x: np.ndarray) -> list[int]:
        hazard = BoostedBoundaryHazard(
            base=ConstantHazard(mean_run_length=period),
            period=period,
            boundary_indices={boundary},
            boost_factor=50,
        )
        model = BOCPD(hazard, BOCPDConfig(max_run_length=period * 2))
        res = model.run(x)
        positions = np.flatnonzero(res.cp_prob >= 0.05)
        out: list[int] = []
        for p in positions:
            if not out or p - out[-1] > 5:
                out.append(int(p))
        return out

    return _run


def run_pelt_binary(penalty: float) -> Callable[[np.ndarray], Sequence[int]]:
    def _run(x: np.ndarray) -> Sequence[int]:
        res = pelt(x, cost_fn=BetaBinomialCost(), penalty=penalty, min_seg_len=5)
        return res.change_points

    return _run


def run_pelt_normal(penalty: float) -> Callable[[np.ndarray], Sequence[int]]:
    def _run(x: np.ndarray) -> Sequence[int]:
        res = pelt(x, cost_fn=NormalMeanVarUnknown(), penalty=penalty, min_seg_len=20)
        return res.change_points

    return _run


def run_edivisive(alpha: float) -> Callable[[np.ndarray], Sequence[int]]:
    def _run(x: np.ndarray) -> Sequence[int]:
        res = edivisive(x, R=10, alpha=alpha, min_size=20, significance=0.2)
        return res.change_points.tolist()

    return _run


def run_hsmm(emission: str) -> Callable[[np.ndarray], Sequence[int]]:
    def _run(x: np.ndarray) -> Sequence[int]:
        x = x[:, None] if x.ndim == 1 else x
        if emission == "diag":
            params = GaussianDiagParams(mu=np.array([[0.0], [3.0]]), var=np.ones((2, 1)))
            L = gaussian_diag_loglik(x, params)
        else:
            params = GaussianFullParams(
                means=np.array([[0.0], [3.0]]), covs=np.array([[[1.0]], [[1.0]]])
            )
            L = gaussian_full_loglik(x, params)
        hsmm_params = HSMMParams(
            pi=np.array([0.5, 0.5]),
            A=np.array([[0.0, 1.0], [1.0, 0.0]]),
            duration=("poisson", PoissonDur(lam=np.array([30.0, 30.0]))),
        )
        model = HSMM(HSMMConfig(K=2, Dmax=50, max_em_iters=1), hsmm_params)
        z, _ = model.decode_viterbi(L)
        cps = np.where(np.diff(z) != 0)[0] + 1
        return cps.tolist()

    return _run


def run_sdhmm(K: int) -> Callable[[np.ndarray], Sequence[int]]:
    def _run(x: np.ndarray) -> Sequence[int]:
        model = SDHMM(SDHMMConfig(K=K, max_iter=30, min_iter=5))
        model.fit(x)
        z = model.viterbi(x)
        cps = np.where(np.diff(z) != 0)[0] + 1
        return cps.tolist()

    return _run


def run_within_period(period: int) -> Callable[[np.ndarray], Sequence[int]]:
    def _run(x: np.ndarray) -> Sequence[int]:
        model = WithinPeriodCPD(ModelPrior(N=period, l=3))
        model.fit(x.astype(bool), RJConfig(iters=100, burn=20))
        return model.result.mode_tau

    return _run


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("snr", [0.6, 0.8])
def test_binary_methods_performance(snr):
    rng = np.random.default_rng(0)
    seg_len = 40
    p0, p1 = 0.5 - snr / 2, 0.5 + snr / 2
    data, cps = binary_data([seg_len, seg_len], [p0, p1], rng)

    methods = {
        "BOCPD": (run_bocpd_const(seg_len), True),
        "PELT": (run_pelt_binary(1.0), False),
    }
    records = {}
    for name, (fn, online) in methods.items():
        metrics = evaluate(fn, data, cps, tol=5, online=online)
        records[name] = metrics
        assert metrics["f1"] >= 0.1
        assert metrics["runtime"] < 1.0
        if online:
            assert metrics["delay"] <= 10
    fig = plot_performance(records, metric="f1")
    plt.close(fig)


@pytest.mark.parametrize("snr", [0.5, 1.5])
@pytest.mark.parametrize("length", [200, 400])
def test_continuous_methods_performance(snr, length):
    rng = np.random.default_rng(1)
    half = length // 2
    data, cps = continuous_data([0.0, snr], [1.0, 1.0], [half, length - half], rng)

    methods = {
        "PELT_p10": run_pelt_normal(10.0),
        "PELT_p20": run_pelt_normal(20.0),
        "ED_alpha1": run_edivisive(1.0),
        "ED_alpha2": run_edivisive(2.0),
    }
    records = {n: evaluate(f, data, cps, tol=5) for n, f in methods.items()}
    # performance thresholds
    assert records["PELT_p10"]["f1"] >= (0.6 if snr < 1.0 else 0.9)
    assert records["ED_alpha1"]["f1"] >= 0.5
    assert records["PELT_p10"]["f1"] >= records["PELT_p20"]["f1"]
    fig = plot_performance(records, metric="f1")
    plt.close(fig)


@pytest.mark.parametrize("period", [20, 30])
def test_periodic_methods_performance(period):
    data, period, tau_true = periodic_data(period, n_cycles=6, cp1=5, cp2=10, break_cycle=3)
    methods = {
        "Within": run_within_period(period),
        "BOCPD": run_bocpd_boost(period, boundary=5),
    }
    records = {}
    for name, fn in methods.items():
        start = time.perf_counter()
        cps = fn(data)
        runtime = time.perf_counter() - start
        taus = sorted(cps) if name == "Within" else sorted({p % period for p in cps})
        prec, rec, f1 = _precision_recall_f1(tau_true, taus, tol=1)
        records[name] = {"precision": prec, "recall": rec, "f1": f1, "runtime": runtime}
        assert f1 >= 0.25
    fig = plot_performance(records, metric="f1")
    plt.close(fig)


@pytest.mark.parametrize("emission", ["diag", "full"])
def test_hsmm_performance(emission):
    rng = np.random.default_rng(3)
    data, cps = continuous_data([0.0, 3.0], [1.0, 1.0], [100, 100], rng)
    metrics = evaluate(run_hsmm(emission), data, cps, tol=2)
    assert metrics["f1"] >= 0.3
    fig = plot_performance({emission: metrics}, metric="f1")
    plt.close(fig)


@pytest.mark.parametrize("K", [2, 3])
def test_sdhmm_performance(K):
    rng = np.random.default_rng(4)
    data, cps = compositional_data(
        [[5.0, 1.0, 1.0], [1.0, 5.0, 1.0]], [80, 80], rng
    )
    metrics = evaluate(run_sdhmm(K), data, cps, tol=2)
    assert metrics["f1"] >= 0.05
    fig = plot_performance({f"SDHMM_K{K}": metrics}, metric="f1")
    plt.close(fig)
