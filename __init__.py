from __future__ import annotations

# Re-export main algorithm classes (clean public API)
from .algorithms.optimization.pelt import PELT
from .algorithms.bayesian.bocpd import BOCPD
from .algorithms.bayesian.within_period import WithinPeriodBOCPD
from .algorithms.nonparametric.edivisive import EDivisive
from .algorithms.state_space.hsmm import HSMM
from .algorithms.kernel.kcp import KernelCPD

# Useful public types
from .core.datatypes import ChangePointResult

__all__ = [
    "PELT",
    "BOCPD",
    "WithinPeriodBOCPD",
    "EDivisive",
    "HSMM",
    "KernelCPD",
    "ChangePointResult",
]

# Attach compatibility layer (lazy attribute fallback + deprecations)
from ._compat import __getattr__, __all__ as _compat_all  # noqa: E402

__all__ += _compat_all
