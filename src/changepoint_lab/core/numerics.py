"""Small numerical stability helpers used by core algorithms."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.floating]


class NumericalStabilityError(FloatingPointError):
    """Raised when an algorithm reaches a non-finite numerical state."""


def require_finite_array(values: Any, name: str) -> ArrayF:
    """Return ``values`` as a float array and reject NaN or infinite entries."""
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise NumericalStabilityError(f"{name} contains non-finite values.")
    return arr


def require_finite_scalar(value: float, name: str) -> float:
    """Return ``value`` as a float and reject NaN or infinite values."""
    out = float(value)
    if not math.isfinite(out):
        raise NumericalStabilityError(f"{name} is not finite.")
    return out


def logsumexp(values: ArrayF, axis: int | None = None) -> ArrayF:
    """Compute ``log(sum(exp(values)))`` while preserving all-``-inf`` slices."""
    arr = np.asarray(values, dtype=float)
    max_value = np.max(arr, axis=axis, keepdims=True)
    finite_max = np.isfinite(max_value)
    shifted = np.where(finite_max, arr - max_value, 0.0)
    total = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    out = np.where(finite_max, max_value + np.log(total), -np.inf)
    if axis is None:
        return np.asarray(out.reshape(()), dtype=float)
    return np.squeeze(out, axis=axis)


def exp_or_inf(log_values: ArrayF) -> ArrayF:
    """Exponentiate finite log-values and map overflow to ``inf``."""
    arr = np.asarray(log_values, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.exp(arr)
    if np.any(np.isnan(out)):
        raise NumericalStabilityError("exp input produced NaN values.")
    return out
