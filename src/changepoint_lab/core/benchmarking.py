"""Reproducible Monte Carlo evaluation of offline changepoint detectors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .comparison import DetectorComparison
from .registry import DetectorRegistry
from .synthetic import simulate_gaussian_segments


@dataclass(frozen=True)
class BenchmarkReplicate:
    """One synthetic replicate with results from all selected detectors."""

    index: int
    comparisons: tuple[DetectorComparison, ...]


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregate detection scores and their Monte Carlo standard errors."""

    detector: str
    repetitions: int
    mean_precision: float
    mean_recall: float
    mean_f1: float
    se_precision: float
    se_recall: float
    se_f1: float


@dataclass(frozen=True)
class GaussianBenchmarkResult:
    """Full replicate records and per-detector summaries."""

    seed: int | None
    tolerance: int
    replicates: tuple[BenchmarkReplicate, ...]
    summaries: tuple[BenchmarkSummary, ...]


def _mean_se(values: Sequence[float] | NDArray[np.floating]) -> tuple[float, float]:
    """Return arithmetic mean and sample-based standard error (zero for n=1)."""
    array = np.asarray(values, dtype=float)
    average = float(np.mean(array))
    if len(array) == 1:
        return average, 0.0
    return average, float(np.std(array, ddof=1) / np.sqrt(len(array)))


def benchmark_gaussian_detectors(
    registry: DetectorRegistry,
    lengths: Sequence[int],
    means: Sequence[float],
    standard_deviations: Sequence[float],
    *,
    repetitions: int,
    tolerance: int = 0,
    seed: int | None = None,
    names: Sequence[str] | None = None,
) -> GaussianBenchmarkResult:
    """Compare fresh detector instances on independent Gaussian simulations.

    The same synthetic sample is evaluated by every detector within each
    replicate. Independent child streams are spawned from ``seed`` for each
    replicate. Summaries contain macro averages across replicates and standard
    errors across those independent replicates; they are not confidence bounds.

    Detector factories must construct independent estimators. If a detector is
    stochastic, its factory is responsible for explicitly controlling its own
    random state; this function seeds only the synthetic observations.
    """
    if not isinstance(repetitions, (int, np.integer)) or isinstance(repetitions, bool):
        raise ValueError("repetitions must be a positive integer")
    if repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if not isinstance(seed, (int, np.integer, type(None))) or isinstance(seed, bool):
        raise ValueError("seed must be an integer or None")

    # Validate detector selection before generating any data.
    selected = registry.names if names is None else tuple(names)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("selected detector names must be nonempty and unique")
    if any(name not in registry.names for name in selected):
        raise KeyError("selected detector name is not registered")

    streams = np.random.SeedSequence(seed).spawn(int(repetitions))
    replicate_records: list[BenchmarkReplicate] = []
    scores_by_detector: dict[str, list[tuple[float, float, float]]] = {
        name: [] for name in selected
    }
    for index, child_seed in enumerate(streams):
        sample = simulate_gaussian_segments(
            lengths, means, standard_deviations, rng=np.random.default_rng(child_seed)
        )
        comparisons = registry.compare(
            sample.values,
            names=selected,
            actual=sample.changepoints,
            tolerance=tolerance,
        )
        replicate_records.append(BenchmarkReplicate(index=index, comparisons=comparisons))
        for comparison in comparisons:
            assert comparison.scores is not None  # Ground truth always supplied.
            scores_by_detector[comparison.name].append(
                (comparison.scores.precision, comparison.scores.recall, comparison.scores.f1)
            )

    summaries: list[BenchmarkSummary] = []
    for name in selected:
        values = np.asarray(scores_by_detector[name], dtype=float)
        mean_precision, se_precision = _mean_se(values[:, 0])
        mean_recall, se_recall = _mean_se(values[:, 1])
        mean_f1, se_f1 = _mean_se(values[:, 2])
        summaries.append(
            BenchmarkSummary(
                detector=name,
                repetitions=int(repetitions),
                mean_precision=mean_precision,
                mean_recall=mean_recall,
                mean_f1=mean_f1,
                se_precision=se_precision,
                se_recall=se_recall,
                se_f1=se_f1,
            )
        )
    return GaussianBenchmarkResult(
        seed=None if seed is None else int(seed),
        tolerance=tolerance,
        replicates=tuple(replicate_records),
        summaries=tuple(summaries),
    )
