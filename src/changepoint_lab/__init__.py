from __future__ import annotations

# ruff: noqa: F401

from ._compat import __getattr__  # noqa: F401
from .api_status import deprecated_symbols, experimental_symbols, stable_symbols
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
from .algorithms.kernel.kcp import KernelCPD, KernelMatrix, RFFConfig
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

__version__ = "0.1.16"

__stable__ = stable_symbols()
__experimental__ = experimental_symbols()
__deprecated__ = deprecated_symbols()

__all__ = [*__stable__, *__experimental__, *__deprecated__]
