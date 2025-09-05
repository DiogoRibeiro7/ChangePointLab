from __future__ import annotations  # ruff: noqa: I001

from ._compat import __all__ as _compat_all
from ._compat import __getattr__  # noqa: F401
from .algorithms.bayesian.bocpd import BOCPD
from .algorithms.bayesian.within_period import WithinPeriodBOCPD
from .algorithms.kernel.kcp import KernelCPD
from .algorithms.nonparametric.edivisive import EDivisive
from .algorithms.optimization.pelt import PELT
from .algorithms.state_space.hsmm import HSMM
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
__all__ += _compat_all
