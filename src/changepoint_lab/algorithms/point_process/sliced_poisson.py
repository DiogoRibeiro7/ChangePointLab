# Scientific traceability:
# - Martinez-Hernandez and Killick (2024), doi:10.1093/biomtc/ujae114.
# - Registry entry: docs/science/method_registry.yml, method id "sliced_poisson_process".

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
import math

import numpy as np
from numpy.typing import NDArray

from ...core.datatypes import SegmentationResult
from ...core.random import make_rng, spawn_rngs
from ...core.segmentation import labels_from_changepoints
from ..optimization.pelt import pelt

ArrayF = NDArray[np.floating[Any]]
ArrayI = NDArray[np.integer[Any]]
Interval = tuple[float, float]
Penalty = Literal["sic", "bic", "aic"]
MarkedMode = Literal["independent", "shared_baseline"]


@dataclass(frozen=True)
class EventPeriod:
    """Event times and observed exposure intervals for one repeated period."""

    event_times: tuple[float, ...]
    exposure_intervals: tuple[Interval, ...] = ()


@dataclass(frozen=True)
class SlicedPoissonConfig:
    """Configuration for sliced Poisson process changepoint detection."""

    period: float = 24.0
    n_basis: int = 5
    degree: int = 3
    min_segment_periods: int = 2
    penalty: float | Penalty | None = "sic"
    quadrature_points: int = 256
    optimizer_max_iter: int = 80
    optimizer_tol: float = 1e-7
    ridge: float = 1e-8
    intensity_floor: float = 1e-9

    def __post_init__(self) -> None:
        if self.period <= 0 or not math.isfinite(self.period):
            raise ValueError("period must be positive and finite.")
        if self.degree < 0:
            raise ValueError("degree must be non-negative.")
        if self.n_basis < self.degree + 1:
            raise ValueError("n_basis must be at least degree + 1.")
        if self.min_segment_periods < 1:
            raise ValueError("min_segment_periods must be positive.")
        if self.quadrature_points < 8:
            raise ValueError("quadrature_points must be at least 8.")
        if self.optimizer_max_iter < 1 or self.optimizer_tol <= 0:
            raise ValueError("optimizer settings must be positive.")
        if self.ridge < 0 or self.intensity_floor <= 0:
            raise ValueError("ridge must be non-negative and intensity_floor positive.")


@dataclass(frozen=True)
class SegmentFit:
    """Maximum-likelihood fit for one contiguous segment of periods."""

    start: int
    end: int
    weights: ArrayF
    cost: float
    neg_log_likelihood: float
    total_events: int
    total_exposure: float
    converged: bool
    iterations: int
    gradient_norm: float
    message: str


@dataclass(frozen=True)
class SlicedPoissonResult:
    """Typed result for sliced Poisson process segmentation."""

    change_points: list[int]
    labels: ArrayI
    total_cost: float
    costs_per_segment: ArrayF
    segment_fits: list[SegmentFit]
    grid_times: ArrayF
    intensity_by_segment: ArrayF
    diagnostics: dict[str, object] = field(default_factory=dict)
    provenance: dict[str, object] = field(default_factory=dict)

    def to_changepoint_result(self) -> SegmentationResult:
        """Return the generic segmentation result view."""
        return SegmentationResult(
            indices=np.asarray(self.change_points, dtype=int),
            score=self.total_cost,
            labels=self.labels,
            method_name="sliced_poisson",
            boundary_convention="right_exclusive",
            objective_orientation="minimize",
            costs_per_segment=self.costs_per_segment,
            metadata={
                "grid_times": self.grid_times,
                "intensity_by_segment": self.intensity_by_segment,
                "diagnostics": self.diagnostics,
            },
            provenance=self.provenance,
        )


@dataclass(frozen=True)
class MarkedSlicedPoissonResult:
    """Explicit marked-process extension result."""

    mode: MarkedMode
    by_mark: Mapping[str, SlicedPoissonResult]
    diagnostics: dict[str, object] = field(default_factory=dict)


def open_uniform_knots(period: float, n_basis: int, degree: int) -> ArrayF:
    """Construct open uniformly spaced knots for a B-spline basis."""
    if n_basis < degree + 1:
        raise ValueError("n_basis must be at least degree + 1.")
    interior_count = n_basis - degree - 1
    interior = (
        np.linspace(0.0, period, interior_count + 2, dtype=float)[1:-1]
        if interior_count > 0
        else np.array([], dtype=float)
    )
    return np.concatenate(
        [
            np.zeros(degree + 1, dtype=float),
            interior,
            np.full(degree + 1, period, dtype=float),
        ]
    )


def bspline_basis(times: Sequence[float] | ArrayF, knots: ArrayF, degree: int) -> ArrayF:
    """Evaluate a B-spline basis via the Cox-de Boor recursion."""
    x = np.asarray(times, dtype=float)
    if x.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    n_basis = len(knots) - degree - 1
    if n_basis <= 0:
        raise ValueError("invalid knot vector for requested degree.")

    basis = np.zeros((x.size, n_basis + degree), dtype=float)
    for idx in range(n_basis + degree):
        left = knots[idx]
        right = knots[idx + 1]
        basis[:, idx] = ((left <= x) & (x < right)).astype(float)
    basis[x == knots[-1], n_basis - 1] = 1.0

    for order in range(1, degree + 1):
        next_basis = np.zeros((x.size, n_basis + degree - order), dtype=float)
        for idx in range(next_basis.shape[1]):
            left_denom = knots[idx + order] - knots[idx]
            right_denom = knots[idx + order + 1] - knots[idx + 1]
            if left_denom > 0:
                next_basis[:, idx] += ((x - knots[idx]) / left_denom) * basis[:, idx]
            if right_denom > 0:
                next_basis[:, idx] += (
                    (knots[idx + order + 1] - x) / right_denom
                ) * basis[:, idx + 1]
        basis = next_basis
    return basis[:, :n_basis]


def normalize_event_periods(
    periods: Sequence[EventPeriod | Sequence[float]],
    period: float,
) -> tuple[EventPeriod, ...]:
    """Validate and normalize event periods to immutable records."""
    normalized: list[EventPeriod] = []
    for item in periods:
        if isinstance(item, EventPeriod):
            event_times = tuple(sorted(float(t) for t in item.event_times))
            exposure = item.exposure_intervals or ((0.0, period),)
        else:
            event_times = tuple(sorted(float(t) for t in item))
            exposure = ((0.0, period),)
        if any(t < 0.0 or t >= period or not math.isfinite(t) for t in event_times):
            raise ValueError("event times must be finite and inside [0, period).")
        clean_exposure = tuple((float(a), float(b)) for a, b in exposure)
        if not clean_exposure:
            raise ValueError("each period must have at least one exposure interval.")
        for start, end in clean_exposure:
            if not (0.0 <= start < end <= period):
                raise ValueError("exposure intervals must satisfy 0 <= start < end <= period.")
        for t in event_times:
            if not any(start <= t < end for start, end in clean_exposure):
                raise ValueError("event times must lie inside observed exposure intervals.")
        normalized.append(EventPeriod(event_times=event_times, exposure_intervals=clean_exposure))
    if not normalized:
        raise ValueError("at least one period is required.")
    return tuple(normalized)


class SlicedPoissonCost:
    """Additive optimized IHPP segment cost for PELT."""

    def __init__(self, periods: tuple[EventPeriod, ...], config: SlicedPoissonConfig):
        self.periods = periods
        self.config = config
        self.knots = open_uniform_knots(config.period, config.n_basis, config.degree)
        self.grid_times = (np.arange(config.quadrature_points, dtype=float) + 0.5) * (
            config.period / config.quadrature_points
        )
        self.grid_basis = bspline_basis(self.grid_times, self.knots, config.degree)
        self.dx = config.period / config.quadrature_points
        self._event_basis_prefix = self._build_event_basis_prefix()
        self._exposure_prefix = self._build_exposure_prefix()
        self._cache: dict[tuple[int, int], SegmentFit] = {}

    def precompute(self, y: ArrayF) -> None:
        """PELT protocol hook; all state is precomputed at construction."""

    def cost(self, a: int, b: int) -> float:
        """Return minus twice optimized log-likelihood for periods [a, b)."""
        return self.fit_segment(a, b).cost

    def fit_segment(self, a: int, b: int) -> SegmentFit:
        """Fit the segment intensity and return diagnostics."""
        if b <= a:
            return SegmentFit(
                start=a,
                end=b,
                weights=np.zeros(self.config.n_basis, dtype=float),
                cost=float("inf"),
                neg_log_likelihood=float("inf"),
                total_events=0,
                total_exposure=0.0,
                converged=False,
                iterations=0,
                gradient_norm=float("inf"),
                message="empty_segment",
            )
        key = (a, b)
        if key not in self._cache:
            self._cache[key] = self._fit_segment_uncached(a, b)
        return self._cache[key]

    def _build_event_basis_prefix(self) -> ArrayF:
        prefix = np.zeros((len(self.periods) + 1, self.config.n_basis), dtype=float)
        for idx, period in enumerate(self.periods):
            if period.event_times:
                basis = bspline_basis(period.event_times, self.knots, self.config.degree)
                prefix[idx + 1] = prefix[idx] + basis.sum(axis=0)
            else:
                prefix[idx + 1] = prefix[idx]
        return prefix

    def _build_exposure_prefix(self) -> ArrayF:
        prefix = np.zeros((len(self.periods) + 1, self.config.quadrature_points), dtype=float)
        for idx, period in enumerate(self.periods):
            weights = np.zeros(self.config.quadrature_points, dtype=float)
            for start, end in period.exposure_intervals:
                weights += ((start <= self.grid_times) & (self.grid_times < end)).astype(float)
            prefix[idx + 1] = prefix[idx] + weights * self.dx
        return prefix

    def _fit_segment_uncached(self, a: int, b: int) -> SegmentFit:
        event_sums = self._event_basis_prefix[b] - self._event_basis_prefix[a]
        exposure_weights = self._exposure_prefix[b] - self._exposure_prefix[a]
        total_events = int(round(float(event_sums.sum())))
        total_exposure = float(exposure_weights.sum())
        if total_exposure <= 0:
            return SegmentFit(
                start=a,
                end=b,
                weights=np.zeros(self.config.n_basis, dtype=float),
                cost=float("inf"),
                neg_log_likelihood=float("inf"),
                total_events=total_events,
                total_exposure=total_exposure,
                converged=False,
                iterations=0,
                gradient_norm=float("inf"),
                message="zero_exposure",
            )
        if total_events == 0:
            weights = np.full(self.config.n_basis, math.log(self.config.intensity_floor))
            nll = total_exposure * self.config.intensity_floor
            return SegmentFit(
                start=a,
                end=b,
                weights=weights,
                cost=2.0 * nll,
                neg_log_likelihood=nll,
                total_events=0,
                total_exposure=total_exposure,
                converged=True,
                iterations=0,
                gradient_norm=0.0,
                message="zero_events_intensity_floor",
            )

        initial_rate = max(total_events / total_exposure, self.config.intensity_floor)
        weights = np.full(self.config.n_basis, math.log(initial_rate), dtype=float)
        converged = False
        message = "max_iter"
        grad_norm = float("inf")
        value = self._objective(weights, exposure_weights, event_sums)
        iterations = 0

        for current_iteration in range(1, self.config.optimizer_max_iter + 1):
            iterations = current_iteration
            value, grad, hess = self._objective_grad_hess(weights, exposure_weights, event_sums)
            grad_norm = float(np.linalg.norm(grad, ord=2))
            if grad_norm < self.config.optimizer_tol:
                converged = True
                message = "converged"
                break
            try:
                step = np.linalg.solve(
                    hess + self.config.ridge * np.eye(self.config.n_basis),
                    -grad,
                )
            except np.linalg.LinAlgError:
                step = -np.linalg.pinv(hess + self.config.ridge * np.eye(self.config.n_basis)) @ grad

            accepted = False
            scale = 1.0
            for _ in range(20):
                candidate = weights + scale * step
                candidate_value = self._objective(candidate, exposure_weights, event_sums)
                if math.isfinite(candidate_value) and candidate_value <= value:
                    weights = candidate
                    value = candidate_value
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                message = "line_search_failed"
                break

        return SegmentFit(
            start=a,
            end=b,
            weights=weights,
            cost=2.0 * value,
            neg_log_likelihood=value,
            total_events=total_events,
            total_exposure=total_exposure,
            converged=converged,
            iterations=iterations,
            gradient_norm=grad_norm,
            message=message,
        )

    def _objective(self, weights: ArrayF, exposure_weights: ArrayF, event_sums: ArrayF) -> float:
        eta = np.clip(self.grid_basis @ weights, -745.0, 80.0)
        return float(np.dot(exposure_weights, np.exp(eta)) - np.dot(event_sums, weights))

    def _objective_grad_hess(
        self,
        weights: ArrayF,
        exposure_weights: ArrayF,
        event_sums: ArrayF,
    ) -> tuple[float, ArrayF, ArrayF]:
        eta = np.clip(self.grid_basis @ weights, -745.0, 80.0)
        weighted_intensity = exposure_weights * np.exp(eta)
        value = float(np.sum(weighted_intensity) - np.dot(event_sums, weights))
        grad = self.grid_basis.T @ weighted_intensity - event_sums
        hess = self.grid_basis.T @ (self.grid_basis * weighted_intensity[:, None])
        return value, grad, hess

    def intensity(self, weights: ArrayF, times: ArrayF) -> ArrayF:
        """Evaluate fitted intensity at supplied times."""
        basis = bspline_basis(times, self.knots, self.config.degree)
        return np.exp(np.clip(basis @ weights, -745.0, 80.0))


@dataclass
class SlicedPoissonCPD:
    """Across-period changepoint detection for sliced IHPP observations."""

    config: SlicedPoissonConfig

    _result: SlicedPoissonResult | None = None

    def fit(self, periods: Sequence[EventPeriod | Sequence[float]]) -> SlicedPoissonCPD:
        """Fit the sliced Poisson process detector."""
        normalized = normalize_event_periods(periods, self.config.period)
        n = len(normalized)
        if self.config.min_segment_periods > n:
            raise ValueError("min_segment_periods cannot exceed number of periods.")
        cost = SlicedPoissonCost(normalized, self.config)
        penalty = _resolve_penalty(self.config.penalty, self.config.n_basis, n)
        pelt_result = pelt(
            np.zeros(n, dtype=float),
            cost,
            penalty=penalty,
            min_seg_len=self.config.min_segment_periods,
            K=0.0,
        )
        edges = [0, *pelt_result.change_points, n]
        segment_fits = [
            cost.fit_segment(start, end)
            for start, end in zip(edges[:-1], edges[1:], strict=True)
        ]
        intensity_by_segment = np.vstack(
            [cost.intensity(segment.weights, cost.grid_times) for segment in segment_fits]
        )
        labels = labels_from_changepoints(
            n,
            pelt_result.change_points,
            min_segment_length=self.config.min_segment_periods,
        ).astype(np.int64)
        failures = [
            {
                "start": segment.start,
                "end": segment.end,
                "message": segment.message,
                "gradient_norm": segment.gradient_norm,
            }
            for segment in segment_fits
            if not segment.converged
        ]
        self._result = SlicedPoissonResult(
            change_points=pelt_result.change_points,
            labels=labels,
            total_cost=float(pelt_result.total_cost),
            costs_per_segment=np.asarray([segment.cost for segment in segment_fits], dtype=float),
            segment_fits=segment_fits,
            grid_times=cost.grid_times,
            intensity_by_segment=intensity_by_segment,
            diagnostics={
                "penalty": penalty,
                "penalty_mode": self.config.penalty,
                "optimization_failures": failures,
                "pelt_objective": (
                    "Cost is optimized minus twice log-likelihood and passed through the shared "
                    "exact PELT objective; K is retained only as a compatibility argument."
                ),
            },
            provenance={
                "method": "sliced_poisson",
                "source": "Martinez-Hernandez and Killick (2024), doi:10.1093/biomtc/ujae114",
                "scope": "faithful_unmarked_ihpp",
                "basis": "open_uniform_b_spline",
            },
        )
        return self

    def predict(self, periods: Sequence[EventPeriod | Sequence[float]] | None = None) -> SlicedPoissonResult:
        """Return the typed sliced Poisson result."""
        if periods is not None:
            return self.fit(periods).predict()
        if self._result is None:
            raise RuntimeError("Call fit before predict.")
        return self._result

    def fit_predict(self, periods: Sequence[EventPeriod | Sequence[float]]) -> SlicedPoissonResult:
        """Fit and return the typed sliced Poisson result."""
        return self.fit(periods).predict()


def _resolve_penalty(penalty: float | Penalty | None, n_basis: int, n_periods: int) -> float:
    if penalty is None or penalty == "sic" or penalty == "bic":
        return (n_basis + 1) * math.log(max(2, n_periods))
    if penalty == "aic":
        return 2.0 * (n_basis + 1)
    value = float(penalty)
    if value < 0 or not math.isfinite(value):
        raise ValueError("penalty must be non-negative and finite.")
    return value


def fit_marked_sliced_poisson(
    periods_by_mark: Mapping[str, Sequence[EventPeriod | Sequence[float]]],
    config: SlicedPoissonConfig,
    *,
    mode: MarkedMode = "independent",
) -> MarkedSlicedPoissonResult:
    """Fit the explicit marked-process extension."""
    if mode == "shared_baseline":
        raise NotImplementedError(
            "shared_baseline marked sliced Poisson is not implemented; use mode='independent'."
        )
    by_mark = {
        mark: SlicedPoissonCPD(config).fit_predict(periods)
        for mark, periods in sorted(periods_by_mark.items())
    }
    return MarkedSlicedPoissonResult(
        mode=mode,
        by_mark=by_mark,
        diagnostics={
            "scope": "mysense_extension",
            "assumption": "marks are fitted as independent sliced Poisson processes",
        },
    )


def simulate_ihpp_periods(
    intensity: Callable[[ArrayF], ArrayF],
    *,
    n_periods: int,
    period: float,
    max_intensity: float,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[tuple[float, ...]]:
    """Simulate IHPP periods by thinning a homogeneous Poisson process."""
    if n_periods < 1 or period <= 0 or max_intensity <= 0:
        raise ValueError("n_periods, period, and max_intensity must be positive.")
    rng = make_rng(seed=seed, rng=rng)
    periods: list[tuple[float, ...]] = []
    for _ in range(n_periods):
        proposal_count = int(rng.poisson(max_intensity * period))
        proposal_times = np.sort(rng.uniform(0.0, period, size=proposal_count))
        if proposal_count == 0:
            periods.append(())
            continue
        rates = np.asarray(intensity(proposal_times), dtype=float)
        if np.any(rates < 0) or np.any(rates > max_intensity * (1.0 + 1e-10)):
            raise ValueError("intensity must stay in [0, max_intensity].")
        keep = rng.random(proposal_count) <= (rates / max_intensity)
        periods.append(tuple(float(t) for t in proposal_times[keep]))
    return periods


def simulate_sliced_poisson_segments(
    segment_lengths: Sequence[int],
    intensities: Sequence[Callable[[ArrayF], ArrayF]],
    *,
    period: float,
    max_intensity: float,
    seed: int | None = None,
) -> tuple[list[tuple[float, ...]], list[int]]:
    """Simulate a sequence of IHPP segments and return true changepoints."""
    if len(segment_lengths) != len(intensities):
        raise ValueError("segment_lengths and intensities must have the same length.")
    rngs = spawn_rngs(seed, len(segment_lengths))
    periods: list[tuple[float, ...]] = []
    change_points: list[int] = []
    running = 0
    for length, intensity, rng in zip(segment_lengths, intensities, rngs, strict=True):
        if length < 1:
            raise ValueError("segment lengths must be positive.")
        periods.extend(
            simulate_ihpp_periods(
                intensity,
                n_periods=length,
                period=period,
                max_intensity=max_intensity,
                rng=rng,
            )
        )
        running += length
        if running < sum(segment_lengths):
            change_points.append(running)
    return periods, change_points


__all__ = [
    "EventPeriod",
    "MarkedSlicedPoissonResult",
    "SlicedPoissonCPD",
    "SlicedPoissonConfig",
    "SlicedPoissonCost",
    "SlicedPoissonResult",
    "bspline_basis",
    "fit_marked_sliced_poisson",
    "normalize_event_periods",
    "open_uniform_knots",
    "simulate_ihpp_periods",
    "simulate_sliced_poisson_segments",
]
