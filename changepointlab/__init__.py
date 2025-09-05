from __future__ import annotations

from warnings import warn

import changepoint_lab
from changepoint_lab import *  # noqa: F401,F403
from changepoint_lab._compat import __all__ as _compat_all
from changepoint_lab._compat import __getattr__  # noqa: F401

warn(
    "`changepointlab` package is deprecated. Use `changepoint_lab`.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = list(changepoint_lab.__all__) + list(_compat_all)
