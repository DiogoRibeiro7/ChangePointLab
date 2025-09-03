import numpy as np

from hsmm.hsmm import HSMM, HSMMConfig, HSMMParams, PoissonDur, HSMMSufficient


def _dummy_suff(cfg: HSMMConfig, T: int) -> HSMMSufficient:
    Dcap = min(cfg.Dmax, T)
    return HSMMSufficient(
        eta=np.zeros((T, cfg.K, Dcap)),
        seg_count=np.zeros(cfg.K),
        seg_total_dur=np.zeros(cfg.K),
        seg_total_d2=np.zeros(cfg.K),
        pi_counts=np.zeros(cfg.K),
        xi_counts=np.zeros((cfg.K, cfg.K)),
        gamma=np.zeros((T, cfg.K)),
        loglik=0.0,
    )


def test_duration_table_cache_invalidation():
    cfg = HSMMConfig(K=2, Dmax=5)
    params = HSMMParams(
        pi=np.array([0.5, 0.5]),
        A=np.full((2, 2), 0.5),
        duration=("poisson", PoissonDur(lam=np.array([2.0, 3.0]))),
    )
    model = HSMM(cfg, params)
    tab1 = model._log_dur_table(10)
    assert 10 in model._dur_cache
    # run M-step which should clear cache
    model._m_step_durations(_dummy_suff(cfg, 10))
    assert model._dur_cache == {}
    tab2 = model._log_dur_table(10)
    assert tab2 is not tab1
