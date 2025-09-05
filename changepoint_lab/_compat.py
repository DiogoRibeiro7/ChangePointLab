from __future__ import annotations

import warnings
from importlib import import_module
from typing import Any

# Map legacy names to their modules/attributes
_LEGACY_ATTRS = {
    "pelt": "pelt.pelt",
    "edivisive": "edivisive.edivisive.edivisive",
    "kcp_penalized": "kcp.kcp.kcp_penalized",
    "kcp_select_bic": "kcp.kcp.kcp_select_bic",
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
