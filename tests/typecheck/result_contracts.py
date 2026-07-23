from __future__ import annotations

import numpy as np

from changepoint_lab.core.datatypes import (
    LatentStateResult,
    OfflineDetector,
    OnlineProbabilityResult,
    PosteriorSampleResult,
    SegmentationResult,
)


def accepts_offline_detector(detector: OfflineDetector[object], x: np.ndarray) -> None:
    result = detector.fit_predict(x)
    indices: np.ndarray = result.indices
    name: str | None = result.method_name
    _ = (indices, name)


def segmentation_fields(result: SegmentationResult) -> None:
    indices: np.ndarray = result.indices
    labels: np.ndarray | None = result.labels
    costs: np.ndarray | None = result.costs_per_segment
    _ = (indices, labels, costs)


def online_fields(result: OnlineProbabilityResult) -> None:
    probabilities: np.ndarray = result.cp_prob
    run_lengths: np.ndarray = result.map_run_length
    _ = (probabilities, run_lengths)


def posterior_fields(result: PosteriorSampleResult) -> None:
    samples: tuple[tuple[int, ...], ...] = result.samples
    log_posteriors: np.ndarray = result.log_posteriors
    _ = (samples, log_posteriors)


def latent_state_fields(result: LatentStateResult) -> None:
    states: np.ndarray = result.states
    durations: np.ndarray | None = result.segment_durations
    _ = (states, durations)

