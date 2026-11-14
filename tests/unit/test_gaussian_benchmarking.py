"""Reproducibility and aggregation tests for Monte Carlo benchmarking."""

import numpy as np
import pytest

from changepoint_lab import (
    ChangePointResult,
    DetectorRegistry,
    benchmark_gaussian_detectors,
)


class FixedBoundaryDetector:
    """A deterministic detector whose boundary is known in advance."""

    def __init__(self, index: int) -> None:
        self.index = index

    def fit_predict(self, data: np.ndarray) -> ChangePointResult:
        return ChangePointResult(indices=np.asarray([self.index]))


def test_repeatable_benchmark_and_fresh_instances() -> None:
    registry = DetectorRegistry()
    instances: list[FixedBoundaryDetector] = []

    def factory() -> FixedBoundaryDetector:
        detector = FixedBoundaryDetector(5)
        instances.append(detector)
        return detector

    registry.register("exact", factory)
    args = ([5, 5], [0.0, 5.0], [1.0, 1.0])
    first = benchmark_gaussian_detectors(registry, *args, repetitions=4, seed=42)
    second = benchmark_gaussian_detectors(registry, *args, repetitions=4, seed=42)
    assert len(instances) == 8
    assert len({id(instance) for instance in instances}) == 8
    assert first.summaries == second.summaries
    assert len(first.replicates) == 4
    assert first.summaries[0].mean_f1 == 1.0
    assert first.summaries[0].se_f1 == 0.0


def test_per_replicate_aggregation_and_standard_error() -> None:
    class DataDependentDetector:
        def fit_predict(self, data: np.ndarray) -> ChangePointResult:
            # Fixed-size series, deliberately choose a true or false boundary.
            return ChangePointResult(indices=np.asarray([5 if data[0] > 0 else 2]))

    registry = DetectorRegistry()
    registry.register("variable", DataDependentDetector)
    result = benchmark_gaussian_detectors(registry, [5, 5], [0, 0], [1, 1], repetitions=20, seed=2)
    f1_values = np.asarray([r.comparisons[0].scores.f1 for r in result.replicates])
    summary = result.summaries[0]
    assert summary.mean_f1 == pytest.approx(float(np.mean(f1_values)))
    assert summary.se_f1 == pytest.approx(float(np.std(f1_values, ddof=1) / np.sqrt(20)))


def test_single_replicate_standard_error_is_zero() -> None:
    registry = DetectorRegistry()
    registry.register("exact", lambda: FixedBoundaryDetector(3))
    result = benchmark_gaussian_detectors(registry, [3, 3], [0, 2], [1, 1], repetitions=1, seed=0)
    assert result.summaries[0].se_f1 == 0.0


@pytest.mark.parametrize("repetitions", [0, -2, 1.5, True])
def test_invalid_repetitions(repetitions) -> None:
    registry = DetectorRegistry()
    registry.register("exact", lambda: FixedBoundaryDetector(2))
    with pytest.raises(ValueError, match="repetitions"):
        benchmark_gaussian_detectors(registry, [2, 2], [0, 1], [1, 1], repetitions=repetitions)


def test_unregistered_selection_is_rejected() -> None:
    registry = DetectorRegistry()
    registry.register("known", lambda: FixedBoundaryDetector(2))
    with pytest.raises(KeyError):
        benchmark_gaussian_detectors(
            registry, [2, 2], [0, 1], [1, 1], repetitions=1, names=["missing"]
        )
