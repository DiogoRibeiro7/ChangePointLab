from __future__ import annotations

import warnings

from changepoint_lab.algorithms.state_space.sdhmm import (
    SDHMM,
    SDHMMConfig,
    SDHMMResult,
)
from changepoint_lab.algorithms.state_space.sdhmm_mix_vi import (
    SDHMMMixVI,
    SDHMMMixVIConfig,
    SDHMMMixVIResult,
)

warnings.warn(
    "`sdhmm` is deprecated; use `changepoint_lab.algorithms.state_space.sdhmm` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "SDHMM",
    "SDHMMConfig",
    "SDHMMResult",
    "SDHMMMixVI",
    "SDHMMMixVIConfig",
    "SDHMMMixVIResult",
]
