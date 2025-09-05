from __future__ import annotations

import warnings
from importlib import import_module
from typing import Any

# Map legacy names to their modules/attributes
_LEGACY_ATTRS = {
    "pelt": "changepoint_lab.algorithms.optimization.pelt.pelt",
    "edivisive": "changepoint_lab.algorithms.nonparametric.edivisive_core.edivisive",
    "hsmm": "changepoint_lab.algorithms.state_space.hsmm_core.HSMM",
    "sdhmm": "changepoint_lab.algorithms.state_space.sdhmm.SDHMM",
    "sdhmm_mix_vi": "changepoint_lab.algorithms.state_space.sdhmm_mix_vi.SDHMMMixVI",
    "kcp_penalized": "changepoint_lab.algorithms.kernel.kcp_core.kcp_penalized",
    "kcp_select_bic": "changepoint_lab.algorithms.kernel.kcp_core.kcp_select_bic",
    "gram_rbf": "changepoint_lab.algorithms.kernel.kcp_core.gram_rbf",
}

__all__ = list(_LEGACY_ATTRS.keys())


def __getattr__(name: str) -> Any:
    if name in _LEGACY_ATTRS:
        warnings.warn(
            f"`{name}` is deprecated; import via the new algorithms namespace",
            DeprecationWarning,
            stacklevel=2,
        )
        module_name, attr = _LEGACY_ATTRS[name].rsplit('.', 1)
        mod = import_module(module_name)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
