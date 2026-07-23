import warnings

import numpy as np
import pytest


def _hsmm_loglik() -> np.ndarray:
    from changepoint_lab.algorithms.state_space.emissions.gaussian_diag import (
        GaussianDiagParams,
        gaussian_diag_loglik,
    )

    obs = np.array([0.0, 0.1, 1.0, 1.1])
    params = GaussianDiagParams(
        mu=np.array([[0.0], [1.0]]),
        var=np.array([[0.2], [0.2]]),
    )
    return gaussian_diag_loglik(obs.reshape(-1, 1), params)


def test_top_level_exports_work_on_tiny_inputs():
    import changepoint_lab as cpl
    from changepoint_lab.algorithms.bayesian.within_period import ModelPrior, RJConfig
    from changepoint_lab.algorithms.optimization.pelt import NormalMeanKnownVar

    pelt = cpl.PELT(
        cost_fn=NormalMeanKnownVar(sigma2=1.0),
        penalty=1.0,
        min_seg_len=2,
    ).fit_predict(np.array([0.0, 0.0, 0.0, 4.0, 4.0, 4.0]))
    assert isinstance(pelt, cpl.SegmentationResult)
    assert pelt.method_name == "pelt"
    assert pelt.boundary_convention == "right_exclusive"
    assert cpl.changepoints_to_edges(pelt.indices, n=6).tolist() == [0, 3, 6]

    bocpd = cpl.BOCPD(
        cpl.ConstantHazard(mean_run_length=4),
        cpl.BOCPDConfig(max_run_length=8, prune_epsilon=0.0, cp_scale=1.0),
    ).fit_predict(np.array([0, 0, 0, 1, 1, 1]))
    assert isinstance(bocpd, cpl.OnlineProbabilityResult)
    assert bocpd.boundary_convention == "time_index"
    assert bocpd.cp_prob.shape == (6,)

    counts = cpl.BOCPD(
        cpl.ConstantHazard(mean_run_length=4),
        cpl.BOCPDConfig(max_run_length=8, prune_epsilon=0.0),
        likelihood=cpl.PoissonGamma(shape0=2.0, rate0=3.0),
    ).run(np.array([0, 1, 3], dtype=int))
    assert counts.cp_prob.shape == (3,)

    edivisive = cpl.EDivisive(min_size=2, R=9, seed=0).fit_predict(
        np.array([0.0, 0.0, 0.1, 1.0, 1.1, 1.2])
    )
    assert isinstance(edivisive, cpl.SegmentationResult)

    kernel = cpl.KernelCPD(penalty=0.1).fit_predict(
        np.array([[0.0], [0.0], [1.0], [1.0]])
    )
    assert isinstance(kernel, cpl.SegmentationResult)

    sliced = cpl.SlicedPoissonCPD(
        cpl.SlicedPoissonConfig(period=1.0, n_basis=1, degree=0, min_segment_periods=2, penalty=1.0)
    ).fit_predict([(0.1,), (0.2,), (0.8,), (0.9,)])
    assert isinstance(sliced, cpl.SlicedPoissonResult)
    assert sliced.to_changepoint_result().boundary_convention == "right_exclusive"

    prior = ModelPrior(N=20, l=5)
    cfg = RJConfig(iters=20, burn=5, thin=5, seed=0)
    within = cpl.WithinPeriodCPD(prior, cfg=cfg).fit_predict(
        np.array([0, 1, 0, 1] * 5, dtype=bool)
    )
    assert isinstance(within, cpl.PosteriorSampleResult)
    assert within.boundary_convention == "periodic_bin_end"
    circular = cpl.CircularChangePoints(period=20, indices=within.indices)
    assert isinstance(circular, cpl.CircularChangePoints)

    hsmm_params = cpl.HSMMParams(
        pi=np.array([1.0, 0.0]),
        A=np.array([[0.0, 1.0], [1.0, 0.0]]),
        duration=("poisson", cpl.PoissonDur(lam=np.array([2.0, 2.0]))),
    )
    hsmm = cpl.HSMM(
        cpl.HSMMConfig(K=2, Dmax=3, max_em_iters=1, learn_durations=False, seed=0),
        hsmm_params,
    ).fit_predict(_hsmm_loglik())
    assert isinstance(hsmm, cpl.LatentStateResult)

    comps = np.array(
        [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.1, 0.9],
        ],
        dtype=float,
    )
    sdhmm = cpl.SDHMM(cpl.SDHMMConfig(K=2, max_iter=2, min_iter=1, seed=0)).fit_predict(comps)
    assert isinstance(sdhmm, cpl.LatentStateResult)

    mix = cpl.SDHMMMixVI(
        cpl.SDHMMMixVIConfig(K=2, M=1, max_iter=2, min_iter=1, seed=0)
    ).fit_predict(comps)
    assert isinstance(mix, cpl.LatentStateResult)


def test_deprecated_imports_warn():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from changepoint_lab import pelt as _  # noqa: F401
        assert any(issubclass(ww.category, DeprecationWarning) for ww in w)


def test_legacy_module_removed():
    with pytest.raises(ModuleNotFoundError):
        import changepointlab  # noqa: F401
