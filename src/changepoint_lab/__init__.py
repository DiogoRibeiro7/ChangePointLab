from __future__ import annotations

from ._compat import __all__ as _compat_all
from ._compat import __getattr__  # noqa: F401
from .algorithms.bayesian.bocpd import (
    BOCPD,
    BOCPDAlertConfig,
    BOCPDConfig,
    BOCPDResult,
    BetaBernoulli,
    BoostedBoundaryHazard,
    ConstantHazard,
    ConjugateLikelihood,
    Hazard,
    PoissonGamma,
    ScheduledHazard,
    extract_changepoint_alerts,
)
from .algorithms.bayesian.within_period import WithinPeriodCPD
from .algorithms.kernel.kcp import KernelCPD
from .algorithms.kernel.kcp_core import gram_rbf, kcp_penalized, kcp_select_bic
from .algorithms.nonparametric.edivisive import EDivisive
from .algorithms.nonparametric.edivisive_core import (
    edivisive,
    EDivisiveResult,
    EDivisiveSplit,
)
from .algorithms.optimization.pelt import PELT
from .algorithms.point_process.sliced_poisson import (
    EventPeriod,
    MarkedSlicedPoissonResult,
    SlicedPoissonCPD,
    SlicedPoissonConfig,
    SlicedPoissonResult,
    fit_marked_sliced_poisson,
)
from .algorithms.state_space.hsmm import HSMM, HSMMConfig, HSMMParams, PoissonDur
from .algorithms.state_space.sdhmm import SDHMM, SDHMMConfig, SDHMMResult
from .algorithms.state_space.sdhmm_mix_vi import (
    SDHMMMixVI,
    SDHMMMixVIConfig,
    SDHMMMixVIResult,
)
from .core.datatypes import (
    ChangePointResult,
    LatentStateDecoder,
    LatentStateResult,
    ModelSelectionResult,
    OfflineDetector,
    OnlineDetector,
    OnlineProbabilityResult,
    PosteriorSampleResult,
    PosteriorSampler,
    SegmentationResult,
)
from .core.segmentation import (
    CircularChangePoints,
    CircularSegment,
    changepoints_from_labels,
    changepoints_to_edges,
    edges_to_changepoints,
    labels_from_changepoints,
    normalize_linear_changepoints,
    segment_slices,
)
from .core.random import choose_from_sequence, make_rng, spawn_rngs

__version__ = "0.1.13"

__all__ = [
    "__version__",
    "PELT",
    "EventPeriod",
    "SlicedPoissonCPD",
    "SlicedPoissonConfig",
    "SlicedPoissonResult",
    "MarkedSlicedPoissonResult",
    "fit_marked_sliced_poisson",
    "BOCPD",
    "BOCPDAlertConfig",
    "BOCPDConfig",
    "BOCPDResult",
    "ConjugateLikelihood",
    "BetaBernoulli",
    "PoissonGamma",
    "Hazard",
    "ConstantHazard",
    "BoostedBoundaryHazard",
    "ScheduledHazard",
    "extract_changepoint_alerts",
    "WithinPeriodCPD",
    "edivisive",
    "EDivisive",
    "EDivisiveResult",
    "EDivisiveSplit",
    "HSMM",
    "HSMMConfig",
    "HSMMParams",
    "PoissonDur",
    "SDHMM",
    "SDHMMConfig",
    "SDHMMResult",
    "SDHMMMixVI",
    "SDHMMMixVIConfig",
    "SDHMMMixVIResult",
    "KernelCPD", 
    "gram_rbf", 
    "kcp_penalized", 
    "kcp_select_bic", 
    "ChangePointResult", 
    "SegmentationResult",
    "OnlineProbabilityResult",
    "PosteriorSampleResult",
    "LatentStateResult",
    "ModelSelectionResult",
    "OfflineDetector",
    "OnlineDetector",
    "LatentStateDecoder",
    "PosteriorSampler",
    "CircularChangePoints",
    "CircularSegment",
    "normalize_linear_changepoints",
    "changepoints_to_edges",
    "edges_to_changepoints",
    "labels_from_changepoints",
    "changepoints_from_labels",
    "segment_slices",
    "make_rng",
    "spawn_rngs",
    "choose_from_sequence",
]

# Attach compatibility layer (lazy attribute fallback + deprecations)
__all__ += _compat_all
