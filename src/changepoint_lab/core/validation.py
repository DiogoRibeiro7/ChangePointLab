from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def require_ndarray(value: Any, *, name: str = "x") -> np.ndarray:
    """Return ``value`` when it is a NumPy array, else raise ``TypeError``."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"`{name}` must be np.ndarray, got {type(value)!r}")
    return value


def validate_array_shape(
    value: np.ndarray,
    *,
    name: str = "x",
    ndim: Sequence[int] = (1, 2),
    non_empty: bool = True,
) -> np.ndarray:
    """Validate array dimensionality without imposing a statistical data domain."""
    if value.ndim not in set(ndim):
        allowed = " or ".join(str(item) for item in ndim)
        raise ValueError(f"`{name}` must be {allowed}D, got {name}.ndim = {value.ndim}")
    if non_empty and value.size == 0:
        raise ValueError(f"`{name}` cannot be empty.")
    return value


def as_finite_float_array(
    value: Any,
    *,
    name: str,
    ndim: Sequence[int] = (1,),
    non_empty: bool = True,
) -> NDArray[np.floating[Any]]:
    """Return a finite float array with the requested dimensionality."""
    array = np.asarray(value, dtype=float)
    validate_array_shape(array, name=name, ndim=ndim, non_empty=non_empty)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"`{name}` must contain only finite values.")
    return array


def as_binary_array(value: Any, *, name: str = "x") -> NDArray[np.bool_]:
    """Return a one-dimensional binary array, accepting only bools or exact 0/1 values."""
    raw = np.asarray(value)
    validate_array_shape(raw, name=name, ndim=(1,), non_empty=True)
    if raw.dtype.kind == "b":
        return np.asarray(raw, dtype=bool)
    numeric = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(numeric)) or np.any((numeric != 0.0) & (numeric != 1.0)):
        raise ValueError(f"`{name}` must contain only binary values 0/1 or bool.")
    return numeric.astype(bool)


def as_count_array(value: Any, *, name: str = "x") -> NDArray[np.integer[Any]]:
    """Return a one-dimensional non-negative integer count array and reject bools."""
    raw = np.asarray(value)
    validate_array_shape(raw, name=name, ndim=(1,), non_empty=True)
    if raw.dtype.kind == "b":
        raise ValueError(f"`{name}` must contain integer counts, not bool values.")
    numeric = np.asarray(value, dtype=float)
    if (
        np.any(~np.isfinite(numeric))
        or np.any(numeric < 0.0)
        or np.any(numeric != np.floor(numeric))
    ):
        raise ValueError(f"`{name}` must contain finite non-negative integer counts.")
    return numeric.astype(int)


def as_probability_array(value: Any, *, name: str = "p") -> NDArray[np.floating[Any]]:
    """Return a one-dimensional probability array with entries in ``[0, 1]``."""
    array = as_finite_float_array(value, name=name, ndim=(1,))
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"`{name}` must contain probabilities in [0, 1].")
    return array


def as_strictly_increasing_times(value: Any, *, name: str = "times") -> NDArray[np.floating[Any]]:
    """Return finite one-dimensional times that are strictly increasing."""
    times = as_finite_float_array(value, name=name, ndim=(1,))
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError(f"`{name}` must be strictly increasing.")
    return times


def as_square_matrix(
    value: Any,
    *,
    name: str,
    symmetric: bool = False,
    psd: bool = False,
    tol: float = 1e-10,
) -> NDArray[np.floating[Any]]:
    """Return a finite square matrix, optionally symmetric positive semidefinite."""
    matrix = as_finite_float_array(value, name=name, ndim=(2,))
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"`{name}` must be square.")
    if symmetric or psd:
        if not np.allclose(matrix, matrix.T, atol=tol, rtol=0.0):
            raise ValueError(f"`{name}` must be symmetric.")
    if psd:
        eigvals = np.linalg.eigvalsh(matrix)
        if np.min(eigvals) < -tol:
            raise ValueError(f"`{name}` must be positive semidefinite.")
    return matrix
