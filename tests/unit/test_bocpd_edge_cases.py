import numpy as np
import pytest

from bocpd.bocpd import (
    BOCPD,
    BOCPDConfig,
    BoostedBoundaryHazard,
    ConstantHazard,
    ScheduledHazard,
)


class MinSegmentHazard:
    """Hazard that suppresses changepoints until a minimum segment length."""

    def __init__(self, min_seg_len: int, base: ConstantHazard | None = None) -> None:
        self.min_seg_len = min_seg_len
        self.base = base or ConstantHazard()

    def prob(self, r: int, t: int) -> float:
        if r < self.min_seg_len - 1:
            return 0.0
        return self.base.prob(r, t)


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


def test_data_exactly_min_segment_length():
    min_len = 4
    h = MinSegmentHazard(min_seg_len=min_len, base=ConstantHazard(mean_run_length=100))
    model = BOCPD(h, BOCPDConfig())
    data = np.zeros(min_len, dtype=int)
    res = model.run(data)
    assert np.allclose(res.cp_prob[: min_len - 1], 0.0)
    assert res.cp_prob[-1] > 0.0
    assert res.map_run_length[-1] == min_len


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


def test_none_input_raises(default_model):
    with pytest.raises(TypeError):
        default_model.run(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("values", [[-1, 2], [0.5, -0.3]])
def test_non_binary_inputs_coerced(values, default_model):
    res = default_model.run(values)
    assert res.cp_prob.shape == (len(values),)
    assert np.isfinite(res.cp_prob).all()
