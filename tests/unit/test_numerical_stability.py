import numpy as np
import pytest


pytestmark = pytest.mark.slow

from bocpd.bocpd import BOCPD, BOCPDConfig, ConstantHazard
from algorithms.optimization.pelt import pelt
from algorithms.optimization.cost_functions import (
    NormalMeanKnownVar,
    NormalMeanVarUnknown,
)
from edivisive.edivisive import edivisive
from hsmm.hsmm import HSMM, HSMMConfig, HSMMParams, PoissonDur
from sdhmm.sdhmm import SDHMM, SDHMMConfig


# ------------------------- BOCPD ---------------------------------

@pytest.mark.parametrize(
    "seq",
    [np.concatenate([[1], np.zeros(199, dtype=int)]),
     np.concatenate([[0], np.ones(199, dtype=int)])],
)
def test_bocpd_sparse_and_dense_sequences(seq):
    cfg = BOCPDConfig(max_run_length=512, store_run_length_posterior=False)
    model = BOCPD(ConstantHazard(mean_run_length=200), cfg)
    for x in seq:
        model.update(int(x))
    assert np.isclose(model.R_prev.sum(), 1.0, rtol=1e-6)
    assert np.all(np.isfinite(model.R_prev))


def test_bocpd_run_length_underflow():
    seq = np.zeros(2000, dtype=int)
    cfg = BOCPDConfig(max_run_length=1000, store_run_length_posterior=False)
    model = BOCPD(ConstantHazard(mean_run_length=500), cfg)
    for x in seq:
        model.update(int(x))
    assert np.isclose(model.R_prev.sum(), 1.0, rtol=1e-6)
    assert model.normalization_issues_ == 0


def test_bocpd_loglikelihood_finite():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 2, size=100)
    model = BOCPD(ConstantHazard(), BOCPDConfig(store_run_length_posterior=False))
    for x in data:
        model.update(int(x))
    logp = np.log(model.lik.predictive_prob(0))
    assert np.all(np.isfinite(logp))


# ------------------------- PELT ---------------------------------

def test_pelt_large_cost_values():
    y = np.linspace(0, 1e6, 50)
    cost = NormalMeanKnownVar(sigma2=1e-12)
    res = pelt(y, cost, penalty=1.0)
    assert np.isfinite(res.total_cost)


def test_pelt_pruning_edge_case():
    y = np.zeros(100)
    cost = NormalMeanVarUnknown()
    res = pelt(y, cost, penalty=1.0, min_seg_len=2)
    assert len(res.change_points) == 0
    assert np.isfinite(res.total_cost)


def test_pelt_long_sequence_many_cps():
    y = np.tile(np.arange(2), 500).astype(float)
    cost = NormalMeanKnownVar(sigma2=1.0)
    res = pelt(y, cost, penalty=0.1, min_seg_len=1)
    assert len(res.change_points) > 10


# ------------------------- E-Divisive ----------------------------

@pytest.mark.parametrize("alpha", [0.1, 2.0])
@pytest.mark.parametrize("resample", ["iid", "block-permutation", "circular-block-bootstrap"])
def test_edivisive_numerical_stability(alpha, resample):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1e-4, size=200)
    res = edivisive(
        X,
        alpha=alpha,
        min_size=10,
        R=10,
        significance=0.5,
        resample=resample,
        seed=0,
    )
    assert res.change_points.size >= 0
    for sp in res.splits:
        assert 0.0 <= sp.pvalue <= 1.0


# ------------------------- HSMM ---------------------------------

@pytest.fixture
def hsmm_model():
    cfg = HSMMConfig(K=2, Dmax=5, max_em_iters=1, seed=0)
    params = HSMMParams(
        pi=np.array([0.6, 0.4]),
        A=np.array([[0.0, 1.0], [1.0, 0.0]]),
        duration=("poisson", PoissonDur(lam=np.array([3.0, 4.0]))),
    )
    return HSMM(cfg, params)


def test_hsmm_forward_backward_stability(hsmm_model):
    L = np.log(np.full((200, 2), 0.5))
    suff = hsmm_model._e_step(L)
    assert np.allclose(suff.gamma.sum(axis=1), 1.0, rtol=1e-6)
    assert np.isfinite(suff.loglik)
    hsmm_model._m_step_transitions(suff)
    assert np.allclose(hsmm_model.params.A.sum(axis=1), 1.0, rtol=1e-6)


def test_hsmm_viterbi_tie_handling(hsmm_model):
    L = np.zeros((20, 2))
    states, durs = hsmm_model.decode_viterbi(L)
    assert states.shape == (20,)
    assert durs.shape == (20,)


# ------------------------- SD-HMM -------------------------------

@pytest.mark.parametrize("seed", [0, 1])
def test_sdhmm_compositional_boundary(seed):
    base = np.array([
        [0.99, 0.005, 0.005],
        [0.005, 0.99, 0.005],
        [0.005, 0.005, 0.99],
    ])
    X = np.repeat(base, 20, axis=0)
    cfg = SDHMMConfig(
        K=3,
        max_iter=1,
        min_iter=1,
        seed=seed,
        lr_alpha=0.001,
        lr_beta=0.001,
        em_steps=1,
    )
    model = SDHMM(cfg)
    res = model.fit(X)
    assert np.allclose(res.gamma.sum(axis=1), 1.0, rtol=1e-6)
    assert np.isfinite(res.loglik)
    for p in res.params:
        assert np.all(p.alpha > 0)
        assert np.all(p.beta > 0)
        assert np.all(np.isfinite(p.alpha))
        assert np.all(np.isfinite(p.beta))
