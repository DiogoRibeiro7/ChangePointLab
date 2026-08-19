from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

import changepoint_lab as cpl
from changepoint_lab import (
    BOCPD,
    BOCPDAlertConfig,
    BOCPDConfig,
    BOCPDResult,
    BetaBernoulli,
    BoostedBoundaryHazard,
    ChangePointResult,
    CircularChangePoints,
    CircularSegment,
    ConjugateLikelihood,
    ConstantHazard,
    EDivisive,
    EDivisiveResult,
    EDivisiveSplit,
    EventPeriod,
    HSMM,
    HSMMConfig,
    HSMMParams,
    Hazard,
    KernelCPD,
    KernelMatrix,
    LatentStateDecoder,
    LatentStateResult,
    MarkedSlicedPoissonResult,
    ModelSelectionResult,
    OfflineDetector,
    OnlineDetector,
    OnlineProbabilityResult,
    PELT,
    PoissonDur,
    PoissonGamma,
    PosteriorSampleResult,
    PosteriorSampler,
    RFFConfig,
    SDHMM,
    SDHMMConfig,
    SDHMMMixVI,
    SDHMMMixVIConfig,
    SDHMMMixVIResult,
    SDHMMResult,
    ScheduledHazard,
    SegmentationResult,
    SlicedPoissonCPD,
    SlicedPoissonConfig,
    SlicedPoissonResult,
    WithinPeriodCPD,
    changepoints_from_labels,
    changepoints_to_edges,
    choose_from_sequence,
    edges_to_changepoints,
    edivisive,
    extract_changepoint_alerts,
    fit_marked_sliced_poisson,
    gram_rbf,
    kcp_penalized,
    kcp_select_bic,
    labels_from_changepoints,
    make_rng,
    normalize_linear_changepoints,
    segment_slices,
    spawn_rngs,
)


version: str = cpl.__version__

stable_classes: tuple[type[Any], ...] = (
    BOCPD,
    BOCPDAlertConfig,
    BOCPDConfig,
    BOCPDResult,
    BetaBernoulli,
    BoostedBoundaryHazard,
    ChangePointResult,
    CircularChangePoints,
    CircularSegment,
    ConjugateLikelihood,
    ConstantHazard,
    EDivisive,
    EDivisiveResult,
    EDivisiveSplit,
    EventPeriod,
    HSMM,
    HSMMConfig,
    HSMMParams,
    Hazard,
    KernelCPD,
    KernelMatrix,
    LatentStateResult,
    MarkedSlicedPoissonResult,
    ModelSelectionResult,
    OnlineProbabilityResult,
    PELT,
    PoissonDur,
    PoissonGamma,
    PosteriorSampleResult,
    RFFConfig,
    SDHMM,
    SDHMMConfig,
    SDHMMMixVI,
    SDHMMMixVIConfig,
    SDHMMMixVIResult,
    SDHMMResult,
    ScheduledHazard,
    SegmentationResult,
    SlicedPoissonCPD,
    SlicedPoissonConfig,
    SlicedPoissonResult,
    WithinPeriodCPD,
)

callables: tuple[Callable[..., Any], ...] = (
    changepoints_from_labels,
    changepoints_to_edges,
    choose_from_sequence,
    edges_to_changepoints,
    edivisive,
    extract_changepoint_alerts,
    fit_marked_sliced_poisson,
    gram_rbf,
    kcp_penalized,
    kcp_select_bic,
    labels_from_changepoints,
    make_rng,
    normalize_linear_changepoints,
    segment_slices,
    spawn_rngs,
)

detector: OfflineDetector[Any]
online: OnlineDetector
decoder: LatentStateDecoder
sampler: PosteriorSampler

legacy_attrs: tuple[Any, ...] = (
    cpl.pelt,
    cpl.hsmm,
    cpl.sdhmm,
    cpl.sdhmm_mix_vi,
    cpl.within_period,
)

result = ChangePointResult(indices=np.array([2], dtype=int), method_name="fixture")
result_dict: dict[str, Any] = result.to_dict()
