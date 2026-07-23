from __future__ import annotations

from ._compat import __all__ as _compat_all
from ._compat import __getattr__  # noqa: F401
from .algorithms.bayesian.bocpd import (
    BOCPD,
    BOCPDConfig,
    BOCPDResult,
    BoostedBoundaryHazard,
    ConstantHazard,
    Hazard,
    ScheduledHazard,
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

__version__ = "0.1.5"

__all__ = [
    "__version__",
    "PELT",
    "BOCPD",
    "BOCPDConfig",
    "BOCPDResult",
    "Hazard",
    "ConstantHazard",
    "BoostedBoundaryHazard",
    "ScheduledHazard",
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
]

# Attach compatibility layer (lazy attribute fallback + deprecations)
__all__ += _compat_all
