from __future__ import annotations

import math

import numpy as np
import pytest

from changepoint_lab.algorithms.point_process.sliced_poisson import (
    EventPeriod,
    SlicedPoissonCPD,
    SlicedPoissonConfig,
    SlicedPoissonCost,
    bspline_basis,
    fit_marked_sliced_poisson,
    open_uniform_knots,
    simulate_sliced_poisson_segments,
)


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
