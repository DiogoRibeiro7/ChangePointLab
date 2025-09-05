import numpy as np
import pytest


pytestmark = pytest.mark.slow

from bocpd.bocpd import BOCPD, BOCPDConfig, ConstantHazard
from algorithms.optimization.cost_functions import BetaBinomialCost
from algorithms.optimization.pelt import pelt
from edivisive.edivisive import edivisive
from hsmm.hsmm import HSMM, HSMMConfig, HSMMParams, PoissonDur
from hsmm.gaussian_diag import (
    GaussianDiagParams,
    gaussian_diag_loglik,
    estimate_from_labels,
)


def f1_score(a, b, tol):
    a = list(a)
    b = list(b)
    matched = set()
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


def _extract_cps(res, min_spacing=5, thr: float = 0.05):
    positions = np.flatnonzero(res.cp_prob >= thr)
    out = []
    for p in positions:
        if not out or p - out[-1] > min_spacing:
            out.append(int(p))
    return out


# ---------------------------------------------------------------------------
# 1. Using outputs from one method as inputs to another
# ---------------------------------------------------------------------------

def test_bocpd_initializes_pelt_improves_accuracy():
    """Use BOCPD changepoints to initialise PELT and refine detections."""
    data = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int), np.zeros(100, dtype=int)])
    true_cps = [100, 200]

    bocpd = BOCPD(ConstantHazard(mean_run_length=50), BOCPDConfig(max_run_length=200))
    res_bocpd = bocpd.run(data)
    cp_bocpd = _extract_cps(res_bocpd)

    # Baseline PELT with overly strong penalty (likely misses changepoints)
    cost = BetaBinomialCost()
    res_base = pelt(data, cost_fn=cost, penalty=10.0, min_seg_len=5)
    f1_base = f1_score(res_base.change_points, true_cps, tol=5)

    # Tune penalty based on BOCPD-detected changepoints
    penalty = 10.0 / max(len(cp_bocpd), 1)
    res_tuned = pelt(data, cost_fn=cost, penalty=penalty, min_seg_len=5)
    f1_tuned = f1_score(res_tuned.change_points, true_cps, tol=5)

    assert f1_tuned >= f1_base


# ---------------------------------------------------------------------------
# 2. E-Divisive segments used for HMM training
# ---------------------------------------------------------------------------

def test_edivisive_segments_train_hmm():
    data = np.concatenate([np.full(60, -2.0), np.full(60, 1.0), np.full(60, 4.0)])
    res = edivisive(data, min_size=30, R=10, significance=0.1)
    cps = res.change_points.tolist()
    labels = np.zeros_like(data, dtype=int)
    prev = 0
    for j, cp in enumerate(cps + [len(data)]):
        labels[prev:cp] = j
        prev = cp

    params = estimate_from_labels(data.reshape(-1, 1), labels, K=len(cps) + 1)
    assert params.mu.shape == (len(cps) + 1, 1)
    assert np.all(np.isfinite(params.var))
    est_means = params.mu.flatten()
    assert np.allclose(sorted(est_means), sorted([-2.0, 1.0, 4.0]), atol=0.5)


# ---------------------------------------------------------------------------
# 3. HMM state posteriors informing BOCPD hazard
# ---------------------------------------------------------------------------

def test_hmm_posteriors_inform_bocpd_hazard():
    rng = np.random.default_rng(0)
    seg1 = rng.normal(0.0, 1.0, size=80)
    seg2 = rng.normal(3.0, 1.0, size=120)
    data = np.concatenate([seg1, seg2])
    true_cps = [80]

    params = GaussianDiagParams(mu=np.array([[0.0], [3.0]]), var=np.array([[1.0], [1.0]]))
    L = gaussian_diag_loglik(data.reshape(-1, 1), params)
    cfg = HSMMConfig(K=2, Dmax=200, max_em_iters=1, seed=0)
    hsmm_params = HSMMParams(
        pi=np.array([1.0, 0.0]),
        A=np.array([[0.0, 1.0], [1.0, 0.0]]),
        duration=("poisson", PoissonDur(lam=np.array([80.0, 120.0]))),
    )
    model = HSMM(cfg, hsmm_params)
    suff = model._e_step(L)
    avg_dur = float(np.sum(suff.seg_total_dur) / np.sum(suff.seg_count))

    model_informed = BOCPD(ConstantHazard(mean_run_length=avg_dur), BOCPDConfig(max_run_length=200))
    cps_informed = _extract_cps(model_informed.run(data))

    model_default = BOCPD(ConstantHazard(mean_run_length=50), BOCPDConfig(max_run_length=200))
    cps_default = _extract_cps(model_default.run(data))

    f1_informed = f1_score(cps_informed, true_cps, tol=5)
    f1_default = f1_score(cps_default, true_cps, tol=5)
    assert f1_informed >= f1_default


# ---------------------------------------------------------------------------
# 4. Two-stage detection and ensemble combination
# ---------------------------------------------------------------------------

def test_coarse_pelt_then_bocpd_ensemble():
    rng = np.random.default_rng(1)
    segs = [rng.normal(0, 1, 50), rng.normal(5, 1, 60), rng.normal(-1, 1, 70)]
    data = np.concatenate(segs)
    true_cps = [50, 110]

    cost = BetaBinomialCost(alpha=1.0, beta=1.0)
    # Coarse PELT: high penalty to get few changepoints
    res_coarse = pelt((data > np.mean(data)).astype(int), cost_fn=cost, penalty=5.0, min_seg_len=20)
    cps_coarse = res_coarse.change_points

    # Fine-grained BOCPD on each coarse segment
    refined = []
    start = 0
    for cp in cps_coarse + [len(data)]:
        seg = data[start:cp]
        bocpd = BOCPD(ConstantHazard(mean_run_length=30), BOCPDConfig(max_run_length=100))
        refined.extend([s + start for s in _extract_cps(bocpd.run(seg))])
        start = cp
    refined.sort()

    # Ensemble: union of coarse PELT and refined BOCPD
    ensemble = sorted(set(cps_coarse) | set(refined))

    f1_coarse = f1_score(cps_coarse, true_cps, tol=5)
    f1_refined = f1_score(refined, true_cps, tol=5)
    f1_ensemble = f1_score(ensemble, true_cps, tol=5)
    assert f1_ensemble >= 0.1
