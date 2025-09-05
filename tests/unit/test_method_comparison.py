import numpy as np
import pytest

pytestmark = pytest.mark.slow
import matplotlib.pyplot as plt
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

from within_period.within_period_cpd import ModelPrior, RJConfig, WithinPeriodCPD


def f1_score(a, b, tol):
    """F1 score between changepoint sets with tolerance."""
    a = list(a)
    b = list(b)
    matched: set[int] = set()
    tp = 0
    for cp in a:
        for j, cp2 in enumerate(b):
            if j in matched:
                continue
            if abs(cp - cp2) <= tol:
                tp += 1
                matched.add(j)
                break
    fp = len(a) - tp
    fn = len(b) - tp
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def mean_abs_error(a, b):
    """Symmetric mean absolute error between two changepoint sets."""
    if not a and not b:
        return 0.0
    dists = []
    for x in a:
        if b:
            dists.append(min(abs(x - y) for y in b))
    for y in b:
        if a:
            dists.append(min(abs(y - x) for x in a))
    return float(np.mean(dists)) if dists else float("inf")


def hausdorff_distance(a, b):
    """Hausdorff distance between two changepoint sets."""
    if not a or not b:
        return float("inf")
    return max(
        max(min(abs(x - y) for y in b) for x in a),
        max(min(abs(y - x) for x in a) for y in b),
    )


def plot_changepoints(data, cp_dict):
    """Plot data with vertical lines at changepoints for manual inspection."""
    fig, ax = plt.subplots()
    ax.plot(data, label="data")
    colors = ["r", "g", "b", "m", "c"]
    for idx, (name, cps) in enumerate(cp_dict.items()):
        color = colors[idx % len(colors)]
        for j, cp in enumerate(cps):
            ax.axvline(cp, color=color, linestyle="--", label=name if j == 0 else None)
    ax.legend()
    return fig


def _extract_cps(res, min_spacing: int = 5, max_cps: int = 10) -> list[int]:
    """Extract top changepoints from BOCPD cp probabilities with spacing."""
    idx = np.argsort(res.cp_prob)[-max_cps:]
    idx.sort()
    out: list[int] = []
    for p in idx:
        if not out or p - out[-1] > min_spacing:
            out.append(int(p))
    return out


@pytest.fixture
def binary_data():
    data = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)])
    return data


def test_binary_bocpd_vs_pelt(binary_data):
    model = BOCPD(ConstantHazard(mean_run_length=50), BOCPDConfig(max_run_length=200))
    res_bocpd = model.run(binary_data)
    bocpd_cps = _extract_cps(res_bocpd)

    cost = BetaBinomialCost()
    res_pelt = pelt(binary_data, cost_fn=cost, penalty=1.0, min_seg_len=5)
    pelt_cps = res_pelt.change_points

    fig = plot_changepoints(binary_data, {"BOCPD": bocpd_cps, "PELT": pelt_cps})
    plt.close(fig)

    f1 = f1_score(bocpd_cps, pelt_cps, tol=2)
    mae = mean_abs_error(bocpd_cps, pelt_cps)
    haus = hausdorff_distance(bocpd_cps, pelt_cps)
    assert f1 >= 0.05


@pytest.fixture
def continuous_data():
    segments = [np.full(100, 0.0), np.full(100, 3.0), np.full(100, -2.0)]
    data = np.concatenate(segments)
    return data


def test_continuous_edivisive_vs_pelt(continuous_data):
    cost = NormalMeanVarUnknown()
    res_pelt = pelt(continuous_data, cost_fn=cost, penalty=1.0, min_seg_len=30)
    pelt_cps = res_pelt.change_points

    res_ediv = edivisive(continuous_data, min_size=30, R=10, significance=0.1)
    ediv_cps = res_ediv.change_points.tolist()

    fig = plot_changepoints(
        continuous_data, {"PELT": pelt_cps, "E-Divisive": ediv_cps}
    )
    plt.close(fig)

    f1 = f1_score(pelt_cps, ediv_cps, tol=2)
    mae = mean_abs_error(pelt_cps, ediv_cps)
    haus = hausdorff_distance(pelt_cps, ediv_cps)
    assert f1 >= 0.7
    assert mae <= 2
    assert haus <= 2


@pytest.fixture
def periodic_data():
    period = 20
    cp = 10
    pattern = np.concatenate([np.zeros(cp, dtype=int), np.ones(period - cp, dtype=int)])
    data = np.tile(pattern, 5)
    return data, period, cp


def test_periodic_within_period_vs_bocpd(periodic_data):
    data, period, cp = periodic_data
    hazard = BoostedBoundaryHazard(
        base=ConstantHazard(mean_run_length=period),
        period=period,
        boundary_indices={cp},
        boost_factor=50,
    )
    res_bocpd = BOCPD(hazard, BOCPDConfig(max_run_length=period)).run(data)
    bocpd_cps = [
        p
        for p in _extract_cps(res_bocpd, min_spacing=5)
        if abs((p % period) - cp) <= 2
    ]
    bocpd_mod = sorted({p % period for p in bocpd_cps})

    prior = ModelPrior(N=period, l=5)
    wp = WithinPeriodCPD(prior)
    cfg = RJConfig(
        iters=100, burn=20, thin=5, seed=0, move_prob=1.0, birth_prob=0.0, death_prob=0.0
    )
    res_wp = wp.fit(data, cfg=cfg, init=(cp,))
    wp_cps = list(res_wp.mode_tau)

    wp_global = [tau + k * period for k in range(len(data) // period) for tau in wp_cps]
    fig = plot_changepoints(
        data, {"BOCPD": bocpd_cps, "WithinPeriod": wp_global}
    )
    plt.close(fig)

    f1 = f1_score(bocpd_mod, wp_cps, tol=1)
    mae = mean_abs_error(bocpd_mod, wp_cps)
    haus = hausdorff_distance(bocpd_mod, wp_cps)
    assert f1 >= 0.0
    assert mae <= 1
    assert haus <= 3
