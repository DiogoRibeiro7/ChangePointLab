from __future__ import annotations

import warnings
from importlib import import_module
from typing import Any

from .api_status import manifest_entry

# Map legacy names to their modules/attributes
_LEGACY_ATTRS = {
    "pelt": "changepoint_lab.algorithms.optimization.pelt.pelt",
    "hsmm": "changepoint_lab.algorithms.state_space.hsmm_core.HSMM",
    "sdhmm": "changepoint_lab.algorithms.state_space.sdhmm.SDHMM",
    "sdhmm_mix_vi": "changepoint_lab.algorithms.state_space.sdhmm_mix_vi.SDHMMMixVI",
    "within_period": "changepoint_lab.algorithms.bayesian.within_period.WithinPeriodCPD",
}

__all__ = list(_LEGACY_ATTRS.keys())


def __getattr__(name: str) -> Any:
    if name in _LEGACY_ATTRS:
        entry = manifest_entry(name)
        removal_version = entry.get("removal_version", "a future release")
        replacement = entry.get("replacement", "the documented replacement")
        warnings.warn(
            (
                f"`{name}` is deprecated and will be removed in "
                f"{removal_version}; use `{replacement}` instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        module_name, attr = _LEGACY_ATTRS[name].rsplit(".", 1)
        mod = import_module(module_name)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
