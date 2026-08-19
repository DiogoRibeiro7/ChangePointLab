from __future__ import annotations

from typing import Literal, TypedDict


LifecycleStatus = Literal["stable", "experimental", "deprecated"]


class ApiSymbol(TypedDict, total=False):
    """Machine-readable lifecycle metadata for a package-level export."""

    name: str
    status: LifecycleStatus
    import_path: str
    replacement: str
    removal_version: str
    note: str


API_MANIFEST: tuple[ApiSymbol, ...] = (
    {"name": "__version__", "status": "stable", "import_path": "changepoint_lab.__version__"},
    {"name": "PELT", "status": "stable", "import_path": "changepoint_lab.PELT"},
    {"name": "EventPeriod", "status": "stable", "import_path": "changepoint_lab.EventPeriod"},
    {
        "name": "SlicedPoissonCPD",
        "status": "stable",
        "import_path": "changepoint_lab.SlicedPoissonCPD",
    },
    {
        "name": "SlicedPoissonConfig",
        "status": "stable",
        "import_path": "changepoint_lab.SlicedPoissonConfig",
    },
    {
        "name": "SlicedPoissonResult",
        "status": "stable",
        "import_path": "changepoint_lab.SlicedPoissonResult",
    },
    {
        "name": "MarkedSlicedPoissonResult",
        "status": "stable",
        "import_path": "changepoint_lab.MarkedSlicedPoissonResult",
    },
    {
        "name": "fit_marked_sliced_poisson",
        "status": "stable",
        "import_path": "changepoint_lab.fit_marked_sliced_poisson",
    },
    {"name": "BOCPD", "status": "stable", "import_path": "changepoint_lab.BOCPD"},
    {
        "name": "BOCPDAlertConfig",
        "status": "stable",
        "import_path": "changepoint_lab.BOCPDAlertConfig",
    },
    {"name": "BOCPDConfig", "status": "stable", "import_path": "changepoint_lab.BOCPDConfig"},
    {"name": "BOCPDResult", "status": "stable", "import_path": "changepoint_lab.BOCPDResult"},
    {
        "name": "ConjugateLikelihood",
        "status": "stable",
        "import_path": "changepoint_lab.ConjugateLikelihood",
    },
    {"name": "BetaBernoulli", "status": "stable", "import_path": "changepoint_lab.BetaBernoulli"},
    {"name": "PoissonGamma", "status": "stable", "import_path": "changepoint_lab.PoissonGamma"},
    {"name": "Hazard", "status": "stable", "import_path": "changepoint_lab.Hazard"},
    {"name": "ConstantHazard", "status": "stable", "import_path": "changepoint_lab.ConstantHazard"},
    {
        "name": "BoostedBoundaryHazard",
        "status": "stable",
        "import_path": "changepoint_lab.BoostedBoundaryHazard",
    },
    {
        "name": "ScheduledHazard",
        "status": "stable",
        "import_path": "changepoint_lab.ScheduledHazard",
    },
    {
        "name": "extract_changepoint_alerts",
        "status": "stable",
        "import_path": "changepoint_lab.extract_changepoint_alerts",
    },
    {
        "name": "WithinPeriodCPD",
        "status": "stable",
        "import_path": "changepoint_lab.WithinPeriodCPD",
    },
    {"name": "edivisive", "status": "stable", "import_path": "changepoint_lab.edivisive"},
    {"name": "EDivisive", "status": "stable", "import_path": "changepoint_lab.EDivisive"},
    {
        "name": "EDivisiveResult",
        "status": "stable",
        "import_path": "changepoint_lab.EDivisiveResult",
    },
    {
        "name": "EDivisiveSplit",
        "status": "stable",
        "import_path": "changepoint_lab.EDivisiveSplit",
    },
    {"name": "HSMM", "status": "stable", "import_path": "changepoint_lab.HSMM"},
    {"name": "HSMMConfig", "status": "stable", "import_path": "changepoint_lab.HSMMConfig"},
    {"name": "HSMMParams", "status": "stable", "import_path": "changepoint_lab.HSMMParams"},
    {"name": "PoissonDur", "status": "stable", "import_path": "changepoint_lab.PoissonDur"},
    {
        "name": "SDHMM",
        "status": "experimental",
        "import_path": "changepoint_lab.SDHMM",
        "note": "Compositional state-space model; validation remains limited.",
    },
    {
        "name": "SDHMMConfig",
        "status": "experimental",
        "import_path": "changepoint_lab.SDHMMConfig",
        "note": "Configuration for experimental SD-HMM support.",
    },
    {
        "name": "SDHMMResult",
        "status": "experimental",
        "import_path": "changepoint_lab.SDHMMResult",
        "note": "Result object for experimental SD-HMM support.",
    },
    {
        "name": "SDHMMMixVI",
        "status": "experimental",
        "import_path": "changepoint_lab.SDHMMMixVI",
        "note": "Mixture VI state-space model; objective validation remains pending.",
    },
    {
        "name": "SDHMMMixVIConfig",
        "status": "experimental",
        "import_path": "changepoint_lab.SDHMMMixVIConfig",
        "note": "Configuration for experimental SDHMMMixVI support.",
    },
    {
        "name": "SDHMMMixVIResult",
        "status": "experimental",
        "import_path": "changepoint_lab.SDHMMMixVIResult",
        "note": "Result object for experimental SDHMMMixVI support.",
    },
    {"name": "KernelCPD", "status": "stable", "import_path": "changepoint_lab.KernelCPD"},
    {"name": "KernelMatrix", "status": "stable", "import_path": "changepoint_lab.KernelMatrix"},
    {"name": "RFFConfig", "status": "stable", "import_path": "changepoint_lab.RFFConfig"},
    {"name": "gram_rbf", "status": "stable", "import_path": "changepoint_lab.gram_rbf"},
    {
        "name": "kcp_penalized",
        "status": "stable",
        "import_path": "changepoint_lab.kcp_penalized",
    },
    {
        "name": "kcp_select_bic",
        "status": "stable",
        "import_path": "changepoint_lab.kcp_select_bic",
    },
    {
        "name": "ChangePointResult",
        "status": "stable",
        "import_path": "changepoint_lab.ChangePointResult",
    },
    {
        "name": "SegmentationResult",
        "status": "stable",
        "import_path": "changepoint_lab.SegmentationResult",
    },
    {
        "name": "OnlineProbabilityResult",
        "status": "stable",
        "import_path": "changepoint_lab.OnlineProbabilityResult",
    },
    {
        "name": "PosteriorSampleResult",
        "status": "stable",
        "import_path": "changepoint_lab.PosteriorSampleResult",
    },
    {
        "name": "LatentStateResult",
        "status": "stable",
        "import_path": "changepoint_lab.LatentStateResult",
    },
    {
        "name": "ModelSelectionResult",
        "status": "stable",
        "import_path": "changepoint_lab.ModelSelectionResult",
    },
    {
        "name": "OfflineDetector",
        "status": "stable",
        "import_path": "changepoint_lab.OfflineDetector",
    },
    {"name": "OnlineDetector", "status": "stable", "import_path": "changepoint_lab.OnlineDetector"},
    {
        "name": "LatentStateDecoder",
        "status": "stable",
        "import_path": "changepoint_lab.LatentStateDecoder",
    },
    {
        "name": "PosteriorSampler",
        "status": "stable",
        "import_path": "changepoint_lab.PosteriorSampler",
    },
    {
        "name": "CircularChangePoints",
        "status": "stable",
        "import_path": "changepoint_lab.CircularChangePoints",
    },
    {
        "name": "CircularSegment",
        "status": "stable",
        "import_path": "changepoint_lab.CircularSegment",
    },
    {
        "name": "normalize_linear_changepoints",
        "status": "stable",
        "import_path": "changepoint_lab.normalize_linear_changepoints",
    },
    {
        "name": "changepoints_to_edges",
        "status": "stable",
        "import_path": "changepoint_lab.changepoints_to_edges",
    },
    {
        "name": "edges_to_changepoints",
        "status": "stable",
        "import_path": "changepoint_lab.edges_to_changepoints",
    },
    {
        "name": "labels_from_changepoints",
        "status": "stable",
        "import_path": "changepoint_lab.labels_from_changepoints",
    },
    {
        "name": "changepoints_from_labels",
        "status": "stable",
        "import_path": "changepoint_lab.changepoints_from_labels",
    },
    {
        "name": "segment_slices",
        "status": "stable",
        "import_path": "changepoint_lab.segment_slices",
    },
    {"name": "make_rng", "status": "stable", "import_path": "changepoint_lab.make_rng"},
    {"name": "spawn_rngs", "status": "stable", "import_path": "changepoint_lab.spawn_rngs"},
    {
        "name": "choose_from_sequence",
        "status": "stable",
        "import_path": "changepoint_lab.choose_from_sequence",
    },
    {
        "name": "pelt",
        "status": "deprecated",
        "import_path": "changepoint_lab.pelt",
        "replacement": "changepoint_lab.algorithms.optimization.pelt.pelt",
        "removal_version": "0.3.0",
    },
    {
        "name": "hsmm",
        "status": "deprecated",
        "import_path": "changepoint_lab.hsmm",
        "replacement": "changepoint_lab.HSMM",
        "removal_version": "0.3.0",
    },
    {
        "name": "sdhmm",
        "status": "deprecated",
        "import_path": "changepoint_lab.sdhmm",
        "replacement": "changepoint_lab.SDHMM",
        "removal_version": "0.3.0",
    },
    {
        "name": "sdhmm_mix_vi",
        "status": "deprecated",
        "import_path": "changepoint_lab.sdhmm_mix_vi",
        "replacement": "changepoint_lab.SDHMMMixVI",
        "removal_version": "0.3.0",
    },
    {
        "name": "within_period",
        "status": "deprecated",
        "import_path": "changepoint_lab.within_period",
        "replacement": "changepoint_lab.WithinPeriodCPD",
        "removal_version": "0.3.0",
    },
)


def symbols_by_status(status: LifecycleStatus) -> tuple[str, ...]:
    """Return manifest symbol names for a lifecycle status."""
    return tuple(item["name"] for item in API_MANIFEST if item["status"] == status)


def stable_symbols() -> tuple[str, ...]:
    """Return stable package-level symbol names."""
    return symbols_by_status("stable")


def experimental_symbols() -> tuple[str, ...]:
    """Return experimental package-level symbol names."""
    return symbols_by_status("experimental")


def deprecated_symbols() -> tuple[str, ...]:
    """Return deprecated package-level symbol names."""
    return symbols_by_status("deprecated")


def manifest_entry(name: str) -> ApiSymbol:
    """Return lifecycle metadata for one symbol."""
    for item in API_MANIFEST:
        if item["name"] == name:
            return item
    raise KeyError(name)
