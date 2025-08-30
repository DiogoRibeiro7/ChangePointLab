import numpy as np
import pytest

from bocpd.bocpd import (
    BOCPD,
    BOCPDConfig,
    BoostedBoundaryHazard,
    ConstantHazard,
    ScheduledHazard,
)


@pytest.fixture
def default_model():
    """Return a BOCPD instance with default configuration."""
    return BOCPD(ConstantHazard(), BOCPDConfig())


def test_empty_input_returns_empty_arrays(default_model):
    res = default_model.run([])
    assert res.cp_prob.size == 0
    assert res.map_run_length.size == 0
    assert res.pred_mean.size == 0


@pytest.mark.parametrize("obs", [0, 1])
def test_single_point_runs(obs, default_model):
    res = default_model.run([obs])
    assert res.cp_prob.shape == (1,)
    assert 0.0 <= res.cp_prob[0] <= 1.0
    # After observing a single datum the run length should be 1
    assert res.map_run_length.tolist() == [1]


def test_sequence_equal_to_max_run_length():
    cfg = BOCPDConfig(max_run_length=5)
    model = BOCPD(ConstantHazard(mean_run_length=100), cfg)
    data = np.zeros(cfg.max_run_length, dtype=int)
    res = model.run(data)
    assert res.map_run_length[-1] == cfg.max_run_length
    assert res.run_length_posterior.shape == (cfg.max_run_length, cfg.max_run_length + 1)


def test_long_sequence_handles_gracefully():
    cfg = BOCPDConfig(max_run_length=50, store_run_length_posterior=False)
    model = BOCPD(ConstantHazard(mean_run_length=1000), cfg)
    data = np.zeros(10001, dtype=int)
    res = model.run(data)
    assert res.cp_prob.shape == (data.size,)
    assert res.map_run_length[-1] == cfg.max_run_length


@pytest.mark.parametrize("mean", [1e-9, 1e6])
def test_extreme_mean_run_length(mean):
    model = BOCPD(ConstantHazard(mean_run_length=mean), BOCPDConfig())
    res = model.run([0, 1, 0])
    assert np.isfinite(res.cp_prob).all()


@pytest.mark.parametrize("alpha,beta", [(1e-9, 1e-9), (1e6, 1e6)])
def test_extreme_alpha_beta(alpha, beta):
    cfg = BOCPDConfig(alpha0=alpha, beta0=beta)
    model = BOCPD(ConstantHazard(), cfg)
    res = model.run([0, 1])
    assert np.isfinite(res.cp_prob).all()


def test_max_run_length_minimum():
    cfg = BOCPDConfig(max_run_length=1)
    model = BOCPD(ConstantHazard(), cfg)
    res = model.run([0, 0, 0])
    assert res.map_run_length.max() <= 1


def test_constant_hazard_near_zero():
    h = ConstantHazard(mean_run_length=1e-9)
    prob = h.prob(0, 0)
    assert np.isclose(prob, 1.0 - 1e-12)


@pytest.mark.parametrize(
    "schedule,period",
    [
        (np.array([1e-12, 1 - 1e-12]), 2),
        (np.full(1000, 1e-3), 1000),
    ],
)
def test_scheduled_hazard_extremes(schedule, period):
    h = ScheduledHazard(schedule=schedule, period=period)
    for t in [0, period - 1]:
        p = h.prob(0, t)
        assert 0.0 < p < 1.0


@pytest.mark.parametrize("boost", [1e-6, 1e6])
def test_boosted_boundary_hazard_extreme(boost):
    base = ConstantHazard(mean_run_length=100)
    h = BoostedBoundaryHazard(base=base, period=5, boundary_indices={0}, boost_factor=boost)
    p0 = h.prob(0, 0)
    p1 = h.prob(0, 1)
    if boost > 1:
        assert p0 > p1
    else:
        assert p0 < p1
    assert 0.0 < p0 < 1.0
    assert 0.0 < p1 < 1.0


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        ConstantHazard(mean_run_length=0).prob(0, 0)
    with pytest.raises(ValueError):
        BOCPD(ConstantHazard(), BOCPDConfig(alpha0=0))
    with pytest.raises(ValueError):
        ScheduledHazard(schedule=np.array([0.1, 0.2]), period=3)
    with pytest.raises(ValueError):
        BoostedBoundaryHazard(
            base=ConstantHazard(), period=5, boundary_indices={5}
        )
