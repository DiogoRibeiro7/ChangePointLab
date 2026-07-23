from __future__ import annotations

import math

import numpy as np
import pytest

from changepoint_lab.algorithms.bayesian.bocpd import (
    BOCPD,
    BOCPDConfig,
    BetaBernoulli,
    ConstantHazard,
    PoissonGamma,
)


def _poisson_gamma_predictive(count: int, shape: float, rate: float) -> float:
    return math.exp(
        math.lgamma(count + shape)
        - math.lgamma(shape)
        - math.lgamma(count + 1.0)
        + shape * math.log(rate / (rate + 1.0))
        + count * math.log(1.0 / (rate + 1.0))
    )


def test_poisson_gamma_predictive_and_updates_match_formula() -> None:
    likelihood = PoissonGamma(shape0=2.0, rate0=3.0)
    likelihood.init_stats(4)

    expected = _poisson_gamma_predictive(4, shape=2.0, rate=3.0)
    assert likelihood.prior_predictive_prob(4) == pytest.approx(expected)
    assert likelihood.predictive_prob(4).tolist() == pytest.approx([expected] * 4)

    likelihood.update_growth(4)
    likelihood.update_cp(4)
    assert likelihood.stats.shape[:3].tolist() == pytest.approx([6.0, 6.0, 6.0])
    assert likelihood.stats.rate[:3].tolist() == pytest.approx([4.0, 4.0, 4.0])

    with pytest.raises(ValueError):
        likelihood.predictive_prob(1.5)


def test_poisson_gamma_works_through_bocpd_public_api() -> None:
    model = BOCPD(
        ConstantHazard(mean_run_length=5),
        BOCPDConfig(max_run_length=8, prune_epsilon=0.0),
        likelihood=PoissonGamma(shape0=2.0, rate0=3.0),
    )
    result = model.run([0, 1, 3, 2])

    assert result.cp_prob.shape == (4,)
    assert np.allclose(result.run_length_posterior.sum(axis=1), 1.0)
    assert result.diagnostics["likelihood"] == "PoissonGamma"
    assert isinstance(model.lik, PoissonGamma)


def test_batch_run_equals_repeated_update() -> None:
    data = [0, 1, 0, 1, 1]
    batch = BOCPD(
        ConstantHazard(mean_run_length=4),
        BOCPDConfig(max_run_length=8, prune_epsilon=0.0),
    ).run(data)

    online = BOCPD(
        ConstantHazard(mean_run_length=4),
        BOCPDConfig(max_run_length=8, prune_epsilon=0.0),
    )
    rows = [online.update(value) for value in data]

    assert [row["cp_prob"] for row in rows] == pytest.approx(batch.cp_prob.tolist())
    assert [row["map_run_length"] for row in rows] == batch.map_run_length.tolist()
    assert [row["pred_mean"] for row in rows] == pytest.approx(batch.pred_mean.tolist())
    assert np.allclose(online.R_prev, batch.run_length_posterior[-1])


def test_checkpoint_resume_equals_uninterrupted_processing() -> None:
    data = [0, 1, 3, 2, 5]
    cfg = BOCPDConfig(max_run_length=8, prune_epsilon=0.0)
    likelihood = PoissonGamma(shape0=2.0, rate0=3.0)
    full = BOCPD(ConstantHazard(mean_run_length=5), cfg, likelihood=likelihood).run(data)

    first = BOCPD(
        ConstantHazard(mean_run_length=5),
        cfg,
        likelihood=PoissonGamma(shape0=2.0, rate0=3.0),
    )
    first_part = first.update_many(data[:2])
    checkpoint = first.state_dict()

    resumed = BOCPD(
        ConstantHazard(mean_run_length=5),
        cfg,
        likelihood=PoissonGamma(shape0=2.0, rate0=3.0),
    )
    resumed.load_state_dict(checkpoint)
    second_part = resumed.update_many(data[2:])

    cp_prob = np.concatenate([first_part.cp_prob, second_part.cp_prob])
    map_run_length = np.concatenate(
        [first_part.map_run_length, second_part.map_run_length]
    )
    pred_mean = np.concatenate([first_part.pred_mean, second_part.pred_mean])
    assert cp_prob.tolist() == pytest.approx(full.cp_prob.tolist())
    assert map_run_length.tolist() == full.map_run_length.tolist()
    assert pred_mean.tolist() == pytest.approx(full.pred_mean.tolist())
    assert np.allclose(resumed.R_prev, full.run_length_posterior[-1])


def test_missing_observation_advances_without_likelihood_update() -> None:
    model = BOCPD(
        ConstantHazard(mean_run_length=4),
        BOCPDConfig(max_run_length=5, prune_epsilon=0.0),
    )
    model.update(1)
    out = model.update(None)

    assert out["cp_prob"] == pytest.approx(0.25)
    assert model.alpha[:3].tolist() == pytest.approx([1.0, 2.0, 2.0])
    assert model.beta[:3].tolist() == pytest.approx([1.0, 1.0, 1.0])

    nan_result = BOCPD(
        ConstantHazard(mean_run_length=4),
        BOCPDConfig(max_run_length=5, prune_epsilon=0.0),
    ).run([1, np.nan, 0])
    assert np.all(np.isfinite(nan_result.cp_prob))


def test_reset_preserves_injected_likelihood_type() -> None:
    model = BOCPD(
        ConstantHazard(mean_run_length=5),
        BOCPDConfig(max_run_length=8, prune_epsilon=0.0),
        likelihood=PoissonGamma(shape0=2.0, rate0=3.0),
    )
    model.update(4)
    model.reset()

    assert isinstance(model.lik, PoissonGamma)
    result = model.run([0, 1])
    assert result.cp_prob.shape == (2,)
    assert isinstance(model.lik, PoissonGamma)


def test_likelihood_state_kind_mismatch_is_rejected() -> None:
    likelihood = BetaBernoulli()
    likelihood.init_stats(3)
    with pytest.raises(ValueError):
        likelihood.load_state_dict({"kind": "PoissonGamma"})
