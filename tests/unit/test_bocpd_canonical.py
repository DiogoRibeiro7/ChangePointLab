from __future__ import annotations

import warnings

import numpy as np
import pytest

from changepoint_lab.algorithms.bayesian.bocpd import (
    BOCPD,
    BOCPDAlertConfig,
    BOCPDConfig,
    ConstantHazard,
    extract_changepoint_alerts,
)


def _reference_beta_bernoulli_bocpd(
    data: list[int],
    *,
    hazard: float,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> np.ndarray:
    """Independent direct Adams-MacKay recursion for tiny Bernoulli streams."""
    probs = np.array([1.0], dtype=float)
    alpha = np.array([alpha0], dtype=float)
    beta = np.array([beta0], dtype=float)
    rows: list[np.ndarray] = []

    for value in data:
        xi = float(value)
        predictive = alpha / (alpha + beta) if xi == 1.0 else beta / (alpha + beta)
        next_probs = np.zeros(probs.size + 1, dtype=float)
        prior_pred = alpha0 / (alpha0 + beta0) if xi == 1.0 else beta0 / (alpha0 + beta0)
        next_probs[0] = float(np.sum(probs * hazard) * prior_pred)
        next_probs[1:] = probs * (1.0 - hazard) * predictive
        next_probs /= next_probs.sum()
        rows.append(next_probs.copy())

        grown_alpha = alpha + xi
        grown_beta = beta + (1.0 - xi)
        alpha = np.concatenate([[alpha0 + xi], grown_alpha])
        beta = np.concatenate([[beta0 + (1.0 - xi)], grown_beta])
        probs = next_probs

    width = len(data) + 1
    posterior = np.zeros((len(data), width), dtype=float)
    for idx, row in enumerate(rows):
        posterior[idx, : row.size] = row
    return posterior


def test_short_bernoulli_posterior_matches_hand_calculation() -> None:
    model = BOCPD(
        ConstantHazard(mean_run_length=4),
        BOCPDConfig(max_run_length=4, prune_epsilon=0.0),
    )
    result = model.run([1, 0])

    assert result.cp_prob.tolist() == pytest.approx([0.25, 1.0 / 3.0])
    assert result.map_run_length.tolist() == [1, 2]
    assert result.run_length_posterior[0, :2].tolist() == pytest.approx([0.25, 0.75])
    assert result.run_length_posterior[1, :3].tolist() == pytest.approx(
        [1.0 / 3.0, 1.0 / 6.0, 0.5]
    )
    assert result.log_evidence.tolist() == pytest.approx(
        [np.log(0.5), np.log(0.375)]
    )


def test_unpruned_posterior_matches_independent_reference() -> None:
    data = [0, 0, 1, 0, 1, 1]
    model = BOCPD(
        ConstantHazard(mean_run_length=5),
        BOCPDConfig(max_run_length=10, prune_epsilon=0.0),
    )
    result = model.run(data)
    reference = _reference_beta_bernoulli_bocpd(data, hazard=0.2)

    assert np.allclose(result.run_length_posterior[:, : reference.shape[1]], reference)
    assert np.allclose(result.run_length_posterior.sum(axis=1), 1.0)
    assert np.allclose(result.approximation_error, 0.0)
    assert result.diagnostics["posterior_is_calibrated"] is True


def test_alert_policy_is_explicit_and_does_not_change_posterior() -> None:
    data = [0, 0, 0, 1, 1, 1]
    base_cfg = BOCPDConfig(max_run_length=8, prune_epsilon=0.0)
    alert_cfg = BOCPDConfig(
        max_run_length=8,
        prune_epsilon=0.0,
        alert_config=BOCPDAlertConfig(
            probability_threshold=0.3,
            require_local_peak=True,
            use_run_length_reset=True,
            min_spacing=2,
        ),
    )

    base_result = BOCPD(ConstantHazard(mean_run_length=4), base_cfg).run(data)
    alert_result = BOCPD(ConstantHazard(mean_run_length=4), alert_cfg).run(data)

    assert np.allclose(base_result.cp_prob, alert_result.cp_prob)
    assert np.array_equal(base_result.map_run_length, alert_result.map_run_length)
    alerts = extract_changepoint_alerts(alert_result, alert_cfg.alert_config)
    assert alerts.tolist() == [3]


def test_wrapper_uses_alert_config_instead_of_hard_coded_threshold() -> None:
    data = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    no_alerts = BOCPD(
        ConstantHazard(mean_run_length=4),
        BOCPDConfig(max_run_length=8, prune_epsilon=0.0),
    ).fit_predict(data)
    configured = BOCPD(
        ConstantHazard(mean_run_length=4),
        BOCPDConfig(
            max_run_length=8,
            prune_epsilon=0.0,
            alert_config=BOCPDAlertConfig(probability_threshold=0.3),
        ),
    ).fit_predict(data)

    assert no_alerts.indices.tolist() == []
    assert configured.indices.tolist() == [3]
    assert "alert_config" in configured.metadata
    assert "log_evidence" in configured.metadata


def test_approximation_modes_preserve_normalization_and_report_error() -> None:
    data = [0, 1, 0, 1, 0, 1, 1, 1]
    exact = BOCPD(
        ConstantHazard(mean_run_length=6),
        BOCPDConfig(max_run_length=12, prune_epsilon=0.0),
    ).run(data)
    approximate = BOCPD(
        ConstantHazard(mean_run_length=6),
        BOCPDConfig(max_run_length=12, prune_epsilon=0.05, top_k=3),
    ).run(data)

    assert np.allclose(approximate.run_length_posterior.sum(axis=1), 1.0)
    assert np.any(approximate.approximation_error > 0.0)
    assert np.all(
        (approximate.approximation_error >= 0.0)
        & (approximate.approximation_error <= 1.0)
    )
    assert np.max(np.abs(exact.cp_prob - approximate.cp_prob)) < 0.2


def test_seeded_alert_simulation_tracks_hazard_tradeoff() -> None:
    rng = np.random.default_rng(123)
    alert_config = BOCPDAlertConfig(
        probability_threshold=0.25,
        require_local_peak=True,
        min_spacing=5,
    )

    def alerts_for(data: np.ndarray, mean_run_length: float) -> np.ndarray:
        cfg = BOCPDConfig(
            max_run_length=100,
            prune_epsilon=0.0,
            alert_config=alert_config,
        )
        result = BOCPD(ConstantHazard(mean_run_length=mean_run_length), cfg).run(data)
        return extract_changepoint_alerts(result, alert_config)

    high_hazard_false_alarms = 0
    low_hazard_false_alarms = 0
    high_hazard_delays: list[int] = []
    low_hazard_delays: list[int] = []

    for _ in range(30):
        stationary = rng.binomial(1, 0.2, size=80).astype(int)
        high_hazard_false_alarms += alerts_for(stationary, 10).size
        low_hazard_false_alarms += alerts_for(stationary, 40).size

        changed = np.concatenate(
            [
                rng.binomial(1, 0.05, size=40),
                rng.binomial(1, 0.9, size=40),
            ]
        ).astype(int)
        for mean_run_length, delays in [
            (10, high_hazard_delays),
            (40, low_hazard_delays),
        ]:
            post_change = alerts_for(changed, mean_run_length)
            post_change = post_change[post_change >= 40]
            delays.append(int(post_change[0] - 40) if post_change.size else 40)

    assert high_hazard_false_alarms > low_hazard_false_alarms
    assert np.mean(high_hazard_delays) < np.mean(low_hazard_delays)


def test_cp_scale_is_deprecated_compatibility_mode() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = BOCPDConfig(max_run_length=8, prune_epsilon=0.0, cp_scale=20.0)
    assert any(item.category is DeprecationWarning for item in caught)

    result = BOCPD(ConstantHazard(mean_run_length=4), cfg).run([0, 0, 0, 1, 1, 1])
    assert result.cp_prob.tolist() == pytest.approx(
        [
            0.8695652174,
            0.8333333333,
            0.830449827,
            0.9129967777,
            0.8366239025,
            0.8308607557,
        ]
    )
    assert result.diagnostics["posterior_is_calibrated"] is False
