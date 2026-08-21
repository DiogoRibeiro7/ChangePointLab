from __future__ import annotations

import math

import numpy as np
import pytest

import changepoint_lab.algorithms.point_process.sliced_poisson as sliced_poisson_module
from changepoint_lab.algorithms.point_process.sliced_poisson import (
    EventPeriod,
    SlicedPoissonCPD,
    SlicedPoissonConfig,
    SlicedPoissonCost,
    bspline_basis,
    fit_marked_sliced_poisson,
    normalize_event_periods,
    open_uniform_knots,
    simulate_sliced_poisson_segments,
)
from changepoint_lab.core.numerics import NumericalStabilityError


def test_bspline_basis_is_nonnegative_and_partitions_unity() -> None:
    knots = open_uniform_knots(period=24.0, n_basis=5, degree=3)
    grid = np.linspace(0.0, 24.0, 101)
    basis = bspline_basis(grid, knots, degree=3)

    assert basis.shape == (101, 5)
    assert np.all(basis >= 0.0)
    assert np.allclose(basis.sum(axis=1), 1.0)


def test_constant_intensity_segment_cost_matches_analytical_mle() -> None:
    periods = [
        (0.1, 0.2),
        (0.3,),
        (),
        (0.2, 0.4),
    ]
    config = SlicedPoissonConfig(
        period=2.0,
        n_basis=1,
        degree=0,
        min_segment_periods=1,
        penalty=0.0,
    )
    cost = SlicedPoissonCost(tuple(EventPeriod(tuple(p), ((0.0, 2.0),)) for p in periods), config)

    fit = cost.fit_segment(0, 4)
    total_events = 5
    total_exposure = 8.0
    expected_nll = total_events - total_events * math.log(total_events / total_exposure)

    assert fit.converged
    assert fit.total_events == total_events
    assert fit.total_exposure == pytest.approx(total_exposure)
    assert fit.cost == pytest.approx(2.0 * expected_nll)


def test_exposure_windows_change_integral_without_counting_unobserved_time() -> None:
    observed_half = [EventPeriod((0.1,), ((0.0, 0.5),)) for _ in range(4)]
    full_period = [EventPeriod((0.1,), ((0.0, 1.0),)) for _ in range(4)]
    config = SlicedPoissonConfig(
        period=1.0,
        n_basis=1,
        degree=0,
        min_segment_periods=1,
        penalty=0.0,
    )

    half_fit = SlicedPoissonCost(tuple(observed_half), config).fit_segment(0, 4)
    full_fit = SlicedPoissonCost(tuple(full_period), config).fit_segment(0, 4)

    assert half_fit.total_exposure == pytest.approx(2.0)
    assert full_fit.total_exposure == pytest.approx(4.0)
    assert math.exp(half_fit.weights[0]) == pytest.approx(2.0)
    assert math.exp(full_fit.weights[0]) == pytest.approx(1.0)


def test_overlapping_nested_and_duplicate_exposure_windows_are_unioned() -> None:
    period = EventPeriod(
        event_times=(0.7, 0.1),
        exposure_intervals=((0.2, 0.4), (0.0, 0.5), (0.2, 0.4), (0.3, 0.8)),
    )

    (normalized,) = normalize_event_periods([period], period=1.0)

    assert normalized.event_times == (0.1, 0.7)
    assert normalized.exposure_intervals == ((0.0, 0.8),)


def test_touching_exposure_windows_form_one_half_open_union() -> None:
    period = EventPeriod(
        event_times=(0.0, 0.25, 0.5, 0.999),
        exposure_intervals=((0.0, 0.25), (0.25, 0.5), (0.5, 1.0)),
    )

    (normalized,) = normalize_event_periods([period], period=1.0)

    assert normalized.exposure_intervals == ((0.0, 1.0),)


def test_exposure_event_membership_uses_half_open_boundaries() -> None:
    normalize_event_periods(
        [EventPeriod(event_times=(0.0,), exposure_intervals=((0.0, 0.5),))],
        period=1.0,
    )
    normalize_event_periods(
        [EventPeriod(event_times=(0.5,), exposure_intervals=((0.0, 0.5), (0.5, 1.0)))],
        period=1.0,
    )

    with pytest.raises(ValueError, match="observed exposure"):
        normalize_event_periods(
            [EventPeriod(event_times=(0.5,), exposure_intervals=((0.0, 0.5),))],
            period=1.0,
        )
    with pytest.raises(ValueError, match=r"\[0, period\)"):
        normalize_event_periods(
            [EventPeriod(event_times=(1.0,), exposure_intervals=((0.0, 1.0),))],
            period=1.0,
        )


def test_total_exposure_counts_union_measure_for_overlapping_windows() -> None:
    periods = (
        EventPeriod(
            event_times=(0.1,),
            exposure_intervals=((0.0, 0.75), (0.25, 1.0), (0.25, 1.0)),
        ),
    )
    config = SlicedPoissonConfig(
        period=1.0,
        n_basis=1,
        degree=0,
        min_segment_periods=1,
        penalty=0.0,
        quadrature_points=32,
    )

    fit = SlicedPoissonCost(periods, config).fit_segment(0, 1)

    assert fit.total_exposure == pytest.approx(1.0)
    assert math.exp(fit.weights[0]) == pytest.approx(1.0)


def test_narrow_exposure_window_contributes_positive_measure() -> None:
    periods = (EventPeriod(event_times=(0.5005,), exposure_intervals=((0.5, 0.501),)),)
    config = SlicedPoissonConfig(
        period=1.0,
        n_basis=1,
        degree=0,
        min_segment_periods=1,
        penalty=0.0,
        quadrature_points=8,
    )

    fit = SlicedPoissonCost(periods, config).fit_segment(0, 1)

    assert fit.converged
    assert fit.total_exposure == pytest.approx(0.001)
    assert math.exp(fit.weights[0]) == pytest.approx(1000.0)


def test_irregular_exposure_windows_match_constant_intensity_oracle() -> None:
    periods = (
        EventPeriod(event_times=(0.05, 0.45), exposure_intervals=((0.0, 0.1), (0.4, 0.7))),
        EventPeriod(event_times=(0.2,), exposure_intervals=((0.15, 0.25), (0.75, 0.9))),
    )
    config = SlicedPoissonConfig(
        period=1.0,
        n_basis=1,
        degree=0,
        min_segment_periods=1,
        penalty=0.0,
        quadrature_points=8,
    )

    fit = SlicedPoissonCost(periods, config).fit_segment(0, 2)

    total_events = 3
    total_exposure = 0.65
    expected_nll = total_events - total_events * math.log(total_events / total_exposure)
    assert fit.total_exposure == pytest.approx(total_exposure)
    assert fit.cost == pytest.approx(2.0 * expected_nll)


def test_exposure_quadrature_converges_with_more_interval_nodes() -> None:
    intervals = ((0.03, 0.17), (0.42, 0.615), (0.72, 0.93))
    theta = np.array([1.2, -0.7, 0.8, -0.2])

    def integral(quadrature_points: int) -> float:
        config = SlicedPoissonConfig(
            period=1.0,
            n_basis=4,
            degree=2,
            min_segment_periods=1,
            quadrature_points=quadrature_points,
        )
        cost = SlicedPoissonCost(
            (EventPeriod(event_times=(), exposure_intervals=intervals),),
            config,
        )
        design = cost._exposure_design_for_segment(0, 1)
        return float(np.dot(design.weights, np.exp(design.basis @ theta)))

    coarse = integral(8)
    fine = integral(32)
    reference = integral(128)

    assert abs(fine - reference) < abs(coarse - reference)


def test_objective_gradient_and_hessian_match_finite_differences() -> None:
    periods = (
        EventPeriod(
            event_times=(0.08, 0.51, 0.83),
            exposure_intervals=((0.0, 0.2), (0.45, 0.6), (0.75, 0.95)),
        ),
    )
    config = SlicedPoissonConfig(
        period=1.0,
        n_basis=3,
        degree=1,
        min_segment_periods=1,
        quadrature_points=32,
    )
    cost = SlicedPoissonCost(periods, config)
    design = cost._exposure_design_for_segment(0, 1)
    event_sums = cost._event_basis_prefix[1] - cost._event_basis_prefix[0]
    theta = np.array([-0.4, 0.25, 0.6])

    _, grad, hess = cost._objective_grad_hess(
        theta,
        design.basis,
        design.weights,
        event_sums,
    )

    eps = 1e-6
    numerical_grad = np.zeros_like(theta)
    numerical_hess = np.zeros_like(hess)
    for idx in range(theta.size):
        step = np.zeros_like(theta)
        step[idx] = eps
        upper_value = cost._objective(theta + step, design.basis, design.weights, event_sums)
        lower_value = cost._objective(theta - step, design.basis, design.weights, event_sums)
        numerical_grad[idx] = (upper_value - lower_value) / (2.0 * eps)

        _, upper_grad, _ = cost._objective_grad_hess(
            theta + step,
            design.basis,
            design.weights,
            event_sums,
        )
        _, lower_grad, _ = cost._objective_grad_hess(
            theta - step,
            design.basis,
            design.weights,
            event_sums,
        )
        numerical_hess[:, idx] = (upper_grad - lower_grad) / (2.0 * eps)

    assert grad == pytest.approx(numerical_grad)
    assert hess == pytest.approx(numerical_hess)


def test_extreme_intensity_fit_stays_finite_with_step_diagnostics() -> None:
    event_times = tuple(float(t) for t in np.linspace(0.5, 0.501, 64, endpoint=False))
    periods = (EventPeriod(event_times=event_times, exposure_intervals=((0.5, 0.501),)),)
    config = SlicedPoissonConfig(
        period=1.0,
        n_basis=1,
        degree=0,
        min_segment_periods=1,
        penalty=0.0,
        quadrature_points=16,
    )

    fit = SlicedPoissonCost(periods, config).fit_segment(0, 1)

    assert fit.converged
    assert np.all(np.isfinite(fit.weights))
    assert math.isfinite(fit.cost)
    assert fit.accepted_step_scale >= 0.0
    assert math.isfinite(fit.hessian_condition)


def test_singular_hessian_uses_deterministic_damping_retry() -> None:
    periods = (
        EventPeriod(event_times=(0.05, 0.06, 0.07), exposure_intervals=((0.0, 0.5),)),
    )
    config = SlicedPoissonConfig(
        period=1.0,
        n_basis=4,
        degree=0,
        min_segment_periods=1,
        penalty=0.0,
        ridge=0.0,
        quadrature_points=16,
    )

    fit = SlicedPoissonCost(periods, config).fit_segment(0, 1)

    assert fit.converged
    assert fit.retries > 0
    assert fit.hessian_condition == float("inf")


def test_failed_segment_raises_by_default() -> None:
    periods = (EventPeriod(event_times=(0.1, 0.2), exposure_intervals=((0.0, 1.0),)),)
    config = SlicedPoissonConfig(
        period=1.0,
        n_basis=2,
        degree=1,
        min_segment_periods=1,
        optimizer_max_iter=1,
    )

    with pytest.raises(NumericalStabilityError, match="optimization failed"):
        SlicedPoissonCost(periods, config).fit_segment(0, 1)


def test_failed_segment_can_be_penalized_as_invalid() -> None:
    periods = (EventPeriod(event_times=(0.1, 0.2), exposure_intervals=((0.0, 1.0),)),)
    config = SlicedPoissonConfig(
        period=1.0,
        n_basis=2,
        degree=1,
        min_segment_periods=1,
        optimizer_max_iter=1,
        optimizer_failure_policy="penalize_invalid",
    )

    fit = SlicedPoissonCost(periods, config).fit_segment(0, 1)

    assert not fit.converged
    assert fit.message == "max_iter"
    assert fit.cost == float("inf")
    assert fit.iterations == 1


def test_retry_policy_attempts_restart_before_penalizing() -> None:
    periods = (EventPeriod(event_times=(0.1, 0.2), exposure_intervals=((0.0, 1.0),)),)
    config = SlicedPoissonConfig(
        period=1.0,
        n_basis=2,
        degree=1,
        min_segment_periods=1,
        optimizer_max_iter=1,
        optimizer_failure_policy="retry",
    )

    fit = SlicedPoissonCost(periods, config).fit_segment(0, 1)

    assert not fit.converged
    assert fit.cost == float("inf")
    assert fit.retries >= 1


def test_forced_line_search_failure_is_not_a_finite_segment_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    periods = (EventPeriod(event_times=(0.1, 0.2), exposure_intervals=((0.0, 1.0),)),)
    config = SlicedPoissonConfig(
        period=1.0,
        n_basis=2,
        degree=1,
        min_segment_periods=1,
        optimizer_failure_policy="penalize_invalid",
    )
    cost = SlicedPoissonCost(periods, config)
    original_objective = cost._objective
    calls = 0

    def failing_candidate_objective(
        weights: np.ndarray,
        exposure_basis: np.ndarray,
        exposure_weights: np.ndarray,
        event_sums: np.ndarray,
    ) -> float:
        nonlocal calls
        calls += 1
        if calls > 1:
            return float("inf")
        return original_objective(weights, exposure_basis, exposure_weights, event_sums)

    monkeypatch.setattr(cost, "_objective", failing_candidate_objective)

    fit = cost.fit_segment(0, 1)

    assert not fit.converged
    assert fit.message == "line_search_failed"
    assert fit.cost == float("inf")
    assert fit.retries > 0


def test_optimizer_failure_policy_is_validated() -> None:
    with pytest.raises(ValueError, match="optimizer_failure_policy"):
        SlicedPoissonConfig(optimizer_failure_policy="ignore")  # type: ignore[arg-type]


def test_exposure_integration_diagnostics_are_reported() -> None:
    periods = [
        EventPeriod(event_times=(0.1,), exposure_intervals=((0.0, 0.2),)),
        EventPeriod(event_times=(0.3,), exposure_intervals=((0.25, 0.5),)),
    ]
    config = SlicedPoissonConfig(
        period=1.0,
        n_basis=1,
        degree=0,
        min_segment_periods=1,
        penalty=0.0,
        quadrature_points=12,
    )

    result = SlicedPoissonCPD(config).fit_predict(periods)

    diagnostics = result.diagnostics["exposure_integration"]
    assert diagnostics["scheme"] == "interval_gauss_legendre"
    assert diagnostics["nodes_per_interval"] == 12
    assert diagnostics["total_quadrature_nodes"] == 24


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_event_totals_are_direct_counts_at_knot_boundaries(degree: int) -> None:
    config = SlicedPoissonConfig(
        period=1.0,
        n_basis=degree + 3,
        degree=degree,
        min_segment_periods=1,
        penalty=0.0,
        optimizer_failure_policy="penalize_invalid",
    )
    knots = open_uniform_knots(config.period, config.n_basis, config.degree)
    event_times = tuple(float(knot) for knot in knots if 0.0 <= knot < 1.0)
    periods = (EventPeriod(event_times=event_times, exposure_intervals=((0.0, 1.0),)),)

    cost = SlicedPoissonCost(periods, config)
    fit = cost.fit_segment(0, 1)

    assert fit.total_events == len(event_times)
    assert int(cost._event_count_prefix[1]) == len(event_times)
    event_sums = cost._event_basis_prefix[1] - cost._event_basis_prefix[0]
    assert float(event_sums.sum()) == pytest.approx(len(event_times))


def test_large_event_totals_stay_exact_in_constant_intensity_fit() -> None:
    event_times = tuple(float(t) for t in np.linspace(0.0, 1.0, 4096, endpoint=False))
    periods = (EventPeriod(event_times=event_times, exposure_intervals=((0.0, 1.0),)),)
    config = SlicedPoissonConfig(
        period=1.0,
        n_basis=1,
        degree=0,
        min_segment_periods=1,
        penalty=0.0,
    )

    fit = SlicedPoissonCost(periods, config).fit_segment(0, 1)

    assert fit.total_events == 4096
    assert math.exp(fit.weights[0]) == pytest.approx(4096.0)


def test_zero_event_branch_uses_direct_integer_counts() -> None:
    periods = (EventPeriod(event_times=(), exposure_intervals=((0.0, 1.0),)),)
    config = SlicedPoissonConfig(
        period=1.0,
        n_basis=3,
        degree=1,
        min_segment_periods=1,
        penalty=0.0,
    )

    cost = SlicedPoissonCost(periods, config)
    fit = cost.fit_segment(0, 1)

    assert cost._event_count_prefix.dtype == np.int64
    assert fit.total_events == 0
    assert fit.converged
    assert fit.message == "zero_events_intensity_floor"


def test_event_count_assertion_detects_basis_partition_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_bspline_basis = sliced_poisson_module.bspline_basis

    def broken_bspline_basis(
        times: list[float] | np.ndarray | tuple[float, ...],
        knots: np.ndarray,
        degree: int,
    ) -> np.ndarray:
        basis = original_bspline_basis(times, knots, degree)
        if len(np.asarray(times, dtype=float)) == 2:
            return 0.5 * basis
        return basis

    monkeypatch.setattr(sliced_poisson_module, "bspline_basis", broken_bspline_basis)
    config = SlicedPoissonConfig(
        period=1.0,
        n_basis=2,
        degree=1,
        min_segment_periods=1,
    )

    with pytest.raises(AssertionError, match="partition-of-unity"):
        SlicedPoissonCost(
            (EventPeriod(event_times=(0.2, 0.8), exposure_intervals=((0.0, 1.0),)),),
            config,
        )


def test_non_overlapping_exposure_windows_are_preserved() -> None:
    period = EventPeriod(
        event_times=(0.1, 0.6),
        exposure_intervals=((0.0, 0.25), (0.5, 0.75)),
    )

    (normalized,) = normalize_event_periods([period], period=1.0)

    assert normalized.exposure_intervals == ((0.0, 0.25), (0.5, 0.75))


def test_amplitude_change_simulation_recovers_known_boundary() -> None:
    periods, true_cps = simulate_sliced_poisson_segments(
        [20, 20],
        [
            lambda t: np.full_like(t, 2.0, dtype=float),
            lambda t: np.full_like(t, 9.0, dtype=float),
        ],
        period=1.0,
        max_intensity=9.0,
        seed=4,
    )
    config = SlicedPoissonConfig(
        period=1.0,
        n_basis=1,
        degree=0,
        min_segment_periods=5,
        penalty=4.0,
    )

    result = SlicedPoissonCPD(config).fit_predict(periods)

    assert true_cps == [20]
    assert result.change_points == [20]
    assert not result.diagnostics["optimization_failures"]


def test_timing_shape_change_is_detected_with_spline_basis() -> None:
    early = [tuple(5.5 + 0.1 * idx for idx in range(6)) for _ in range(8)]
    late = [tuple(18.0 + 0.1 * idx for idx in range(6)) for _ in range(8)]
    config = SlicedPoissonConfig(
        period=24.0,
        n_basis=6,
        degree=3,
        min_segment_periods=4,
        penalty=5.0,
    )

    result = SlicedPoissonCPD(config).fit_predict([*early, *late])

    assert result.change_points == [8]
    first_peak = float(result.grid_times[np.argmax(result.intensity_by_segment[0])])
    second_peak = float(result.grid_times[np.argmax(result.intensity_by_segment[1])])
    assert first_peak < second_peak


def test_marked_extension_fits_independent_processes_and_rejects_shared_baseline() -> None:
    periods = {
        "door": [(0.1,), (0.2,), (0.8, 0.9), (0.85,)],
        "kettle": [(0.3,), (0.35,), (0.4,), (0.45,)],
    }
    config = SlicedPoissonConfig(
        period=1.0,
        n_basis=1,
        degree=0,
        min_segment_periods=2,
        penalty=1.0,
    )

    result = fit_marked_sliced_poisson(periods, config, mode="independent")

    assert result.mode == "independent"
    assert set(result.by_mark) == {"door", "kettle"}
    with pytest.raises(NotImplementedError, match="shared_baseline"):
        fit_marked_sliced_poisson(periods, config, mode="shared_baseline")
