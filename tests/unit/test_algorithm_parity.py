import numpy as np
from changepoint_lab import BOCPD, PELT, EDivisive
from changepoint_lab import edivisive as legacy_edivisive
from changepoint_lab import pelt as legacy_pelt
from changepoint_lab.algorithms.bayesian.bocpd import BOCPDConfig, ConstantHazard
from changepoint_lab.algorithms.optimization.cost_functions import NormalMeanVarUnknown
from changepoint_lab.algorithms.state_space.hsmm import HSMM, HSMMConfig, HSMMParams

from bocpd.bocpd import BOCPD as LegacyBOCPD
from hsmm.gaussian_diag import GaussianDiagParams, gaussian_diag_loglik
from hsmm.hsmm import PoissonDur


def test_pelt_parity():
    data = np.concatenate([np.zeros(50), np.ones(50)])
    cost = NormalMeanVarUnknown()
    new_res = PELT(cost_fn=cost, penalty=1.0, min_seg_len=10).fit_predict(data)
    old_res = legacy_pelt(data, cost_fn=cost, penalty=1.0, min_seg_len=10)
    assert new_res.indices.tolist() == old_res.change_points


def test_edivisive_parity():
    data = np.concatenate([np.zeros(30), np.ones(30)])
    new_res = EDivisive(min_size=10, R=9, seed=0).fit_predict(data)
    old_res = legacy_edivisive(data, min_size=10, R=9, seed=0)
    assert new_res.indices.tolist() == old_res.change_points.tolist()


def test_bocpd_parity():
    data = np.concatenate([np.zeros(30, dtype=int), np.ones(30, dtype=int)])
    cfg = BOCPDConfig(max_run_length=50)
    model_new = BOCPD(ConstantHazard(mean_run_length=20), cfg)
    res_new = model_new.run(data)
    model_old = LegacyBOCPD(ConstantHazard(mean_run_length=20), cfg)
    res_old = model_old.run(data)
    assert np.allclose(res_new.cp_prob, res_old.cp_prob)


def test_hsmm_parity():
    obs = np.concatenate([np.zeros(30), np.ones(30)])
    params = GaussianDiagParams(mu=np.array([[0.0], [1.0]]), var=np.array([[1.0], [1.0]]))
    loglik = gaussian_diag_loglik(obs.reshape(-1, 1), params)
    cfg = HSMMConfig(K=2, Dmax=40, max_em_iters=1, seed=0)
    hsmm_params = HSMMParams(
        pi=np.array([1.0, 0.0]),
        A=np.array([[0.0, 1.0], [1.0, 0.0]]),
        duration=("poisson", PoissonDur(lam=np.array([30.0, 30.0]))),
    )
    model_new = HSMM(cfg, hsmm_params)
    res_new = model_new.fit_predict(loglik)
    model_old = HSMM(cfg, hsmm_params)
    states_old, durs_old = model_old.decode_viterbi(loglik)
    cps_old = np.cumsum(durs_old)[:-1]
    assert res_new.indices.tolist() == cps_old.tolist()
