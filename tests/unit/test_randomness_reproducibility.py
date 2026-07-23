from __future__ import annotations

import numpy as np

from changepoint_lab import EDivisive
from changepoint_lab.algorithms.bayesian.within_period import (
    ModelPrior,
    RJConfig,
    WithinPeriodCPD,
    WithinPeriodCore,
)
from changepoint_lab.core.random import make_rng, spawn_rngs


def _periodic_stream() -> np.ndarray:
    return np.array([0] * 40 + [1] * 40, dtype=bool)


def test_make_rng_rejects_ambiguous_seed_and_generator() -> None:
    rng = np.random.default_rng(0)
    try:
        make_rng(seed=0, rng=rng)
    except ValueError as exc:
        assert "either seed or rng" in str(exc)
    else:
        raise AssertionError("make_rng accepted both seed and rng")


def test_spawned_streams_are_reproducible_and_independent() -> None:
    left_a, right_a = spawn_rngs(11, 2)
    left_b, right_b = spawn_rngs(11, 2)

    left_draw = left_a.integers(0, 1_000_000, size=16)
    right_draw = right_a.integers(0, 1_000_000, size=16)

    assert np.array_equal(left_draw, left_b.integers(0, 1_000_000, size=16))
    assert np.array_equal(right_draw, right_b.integers(0, 1_000_000, size=16))
    assert not np.array_equal(left_draw, right_draw)


def test_within_period_same_seed_replays_samples() -> None:
    cfg = RJConfig(iters=60, burn=10, thin=5, seed=17)
    model = WithinPeriodCore(ModelPrior(N=20, l=5))
    first = model.fit(_periodic_stream(), cfg=cfg)
    second = WithinPeriodCore(ModelPrior(N=20, l=5)).fit(_periodic_stream(), cfg=cfg)

    assert first.samples_tau == second.samples_tau
    assert first.log_posteriors == second.log_posteriors
    assert first.changepoint_hist.tolist() == second.changepoint_hist.tolist()
    assert first.provenance["seed"] == 17


def test_within_period_repeated_seeded_wrapper_fit_replays() -> None:
    detector = WithinPeriodCPD(
        ModelPrior(N=20, l=5),
        cfg=RJConfig(iters=50, burn=10, thin=5, seed=23),
    )

    first = detector.fit(_periodic_stream()).predict()
    second = detector.fit(_periodic_stream()).predict()

    assert first.samples == second.samples
    assert first.changepoint_hist.tolist() == second.changepoint_hist.tolist()
    assert first.provenance["seed"] == 23


def test_explicit_generator_is_stateful_but_fresh_generators_replay() -> None:
    cfg = RJConfig(iters=80, burn=10, thin=5, seed=None)
    model = WithinPeriodCore(ModelPrior(N=20, l=5))
    stream = _periodic_stream()

    shared = np.random.default_rng(31)
    first = model.fit(stream, cfg=cfg, rng=shared)
    second = model.fit(stream, cfg=cfg, rng=shared)
    replay = model.fit(stream, cfg=cfg, rng=np.random.default_rng(31))

    assert first.samples_tau == replay.samples_tau
    assert first.changepoint_hist.tolist() == replay.changepoint_hist.tolist()
    assert first.samples_tau != second.samples_tau


def test_posterior_summaries_use_explicit_local_rng() -> None:
    model = WithinPeriodCore(ModelPrior(N=20, l=5))
    model._prepare_counts(_periodic_stream())
    samples = [(5, 10), (5, 10), (5, 15)]

    first = model.pointwise_posterior_summary_from_samples(samples, seed=101)
    replay = model.pointwise_posterior_summary_from_samples(samples, seed=101)
    other = model.pointwise_posterior_summary_from_samples(samples, seed=102)

    assert np.allclose(first["median"], replay["median"])
    assert np.allclose(first["lower"], replay["lower"])
    assert not np.allclose(first["median"], other["median"])


def test_stochastic_fit_does_not_perturb_legacy_numpy_global_state() -> None:
    np.random.seed(1234)
    before = np.random.random(4)

    np.random.seed(1234)
    WithinPeriodCore(ModelPrior(N=20, l=5)).fit(
        _periodic_stream(),
        cfg=RJConfig(iters=40, burn=10, thin=5, seed=5),
    )
    after = np.random.random(4)

    assert np.array_equal(before, after)


def test_edivisive_result_exposes_reproducibility_provenance() -> None:
    series = np.array([0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0])

    result = EDivisive(min_size=2, R=9, seed=7).fit_predict(series)

    assert result.provenance["seed"] == 7
    assert result.provenance["rng"] == "numpy.random.Generator"
    assert result.metadata["provenance"] == result.provenance
