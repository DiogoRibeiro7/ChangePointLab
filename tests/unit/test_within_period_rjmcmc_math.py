from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from changepoint_lab.algorithms.bayesian.within_period import ModelPrior, RJConfig
from changepoint_lab.algorithms.bayesian.within_period.within_period_cpd import (
    Tau,
    WithinPeriodCore,
)


def _circular_lengths(tau: Tau, period: int) -> list[int]:
    if not tau:
        return [period]
    lengths: list[int] = []
    previous = tau[-1]
    for boundary in tau:
        length = (boundary - previous) % period
        lengths.append(period if length == 0 else length)
        previous = boundary
    return lengths


def _valid_states(period: int, min_segment_length: int) -> list[Tau]:
    states: list[Tau] = [()]
    max_segments = period // min_segment_length
    for segment_count in range(2, max_segments + 1):
        for tau in itertools.combinations(range(period), segment_count):
            if all(length >= min_segment_length for length in _circular_lengths(tau, period)):
                states.append(tau)
    return states


def _prepared_model(*, pois_lambda: float = 1.0) -> WithinPeriodCore:
    model = WithinPeriodCore(ModelPrior(N=6, l=2, gamma=1.0, pois_lambda=pois_lambda))
    model._prepare_counts(np.array([0, 1, 1, 0, 0, 1] * 2, dtype=bool))
    return model


def _posterior_weights(model: WithinPeriodCore, states: list[Tau]) -> dict[Tau, float]:
    log_posts = {state: model._log_posterior_tau(state) for state in states}
    normalizer = sum(math.exp(value) for value in log_posts.values())
    return {state: math.exp(log_posts[state]) / normalizer for state in states}


def test_rjconfig_rejects_invalid_proposal_probabilities() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        RJConfig(move_prob=-0.1, birth_prob=0.6, death_prob=0.5)
    with pytest.raises(ValueError, match="must equal 1"):
        RJConfig(move_prob=0.2, birth_prob=0.2, death_prob=0.2)


def test_exact_proposal_support_is_reversible_on_tiny_state_space() -> None:
    model = _prepared_model()
    cfg = RJConfig(iters=20, burn=5, thin=1, seed=0)
    states = _valid_states(6, 2)

    for source in states:
        total = sum(step.probability for step in model.proposal_steps(source, cfg))
        assert total == pytest.approx(1.0)
        for target in states:
            q_forward = model.proposal_probability(source, target, cfg)
            q_reverse = model.proposal_probability(target, source, cfg)
            assert (q_forward > 0.0) == (q_reverse > 0.0)


def test_detailed_balance_holds_numerically_on_tiny_state_space() -> None:
    model = _prepared_model(pois_lambda=1.5)
    cfg = RJConfig(iters=20, burn=5, thin=1, seed=0)
    states = _valid_states(6, 2)
    weights = _posterior_weights(model, states)

    for source in states:
        source_log = model._log_posterior_tau(source)
        for target in states:
            if source == target:
                continue
            q_forward = model.proposal_probability(source, target, cfg)
            q_reverse = model.proposal_probability(target, source, cfg)
            if q_forward == 0.0 and q_reverse == 0.0:
                continue
            target_log = model._log_posterior_tau(target)
            alpha_forward = min(
                1.0,
                math.exp(target_log - source_log) * q_reverse / q_forward,
            )
            alpha_reverse = min(
                1.0,
                math.exp(source_log - target_log) * q_forward / q_reverse,
            )

            lhs = weights[source] * q_forward * alpha_forward
            rhs = weights[target] * q_reverse * alpha_reverse
            assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12)


def test_tiny_empirical_stationary_frequencies_match_exact_posterior() -> None:
    model = _prepared_model(pois_lambda=1.5)
    cfg = RJConfig(iters=60_000, burn=10_000, thin=25, seed=12)
    states = _valid_states(6, 2)
    expected = _posterior_weights(model, states)

    result = model.fit(np.array([0, 1, 1, 0, 0, 1] * 2, dtype=bool), cfg=cfg)
    counts = {state: result.samples_tau.count(state) for state in states}
    total = len(result.samples_tau)

    for state in states:
        observed = counts[state] / total
        assert observed == pytest.approx(expected[state], abs=0.07)


def test_non_unit_poisson_lambda_changes_segment_count_prior_odds() -> None:
    low = _prepared_model(pois_lambda=0.5)
    high = _prepared_model(pois_lambda=3.0)
    one_segment: Tau = ()
    two_segments: Tau = (0, 3)

    low_odds = low._log_posterior_tau(two_segments) - low._log_posterior_tau(one_segment)
    high_odds = high._log_posterior_tau(two_segments) - high._log_posterior_tau(one_segment)

    assert high_odds - low_odds == pytest.approx(math.log(3.0 / 0.5))


def test_log_posterior_is_rotation_invariant_with_rotated_data_and_tau() -> None:
    x = np.array([0, 1, 1, 0, 0, 1] * 3, dtype=bool)
    tau: Tau = (0, 3)
    shift = 2

    model = WithinPeriodCore(ModelPrior(N=6, l=2))
    model._prepare_counts(x)
    rotated = WithinPeriodCore(ModelPrior(N=6, l=2))
    rotated._prepare_counts(np.roll(x.reshape(-1, 6), shift, axis=1).ravel())
    rotated_tau = tuple(sorted((boundary + shift) % 6 for boundary in tau))

    assert rotated._log_posterior_tau(rotated_tau) == pytest.approx(
        model._log_posterior_tau(tau)
    )
