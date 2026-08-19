from __future__ import annotations
"""Utility routines for validating BOCPD parameters.

The BOCPD module exposes several configuration and hazard classes which share
similar constraints on their arguments.  Consolidating checks here keeps the
validation logic consistent and makes it easier to extend in the future.
"""
from numbers import Integral, Real
from typing import SupportsFloat, SupportsInt


def positive_float(value: SupportsFloat, name: str) -> float:
    """Return ``value`` as ``float`` if strictly positive, else raise ``ValueError``."""
    if not isinstance(value, Real) or float(value) <= 0.0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def probability(value: SupportsFloat, name: str) -> float:
    """Return ``value`` as ``float`` if in (0,1), else raise ``ValueError``."""
    v = float(value)
    if not isinstance(value, Real) or not (0.0 < v < 1.0):
        raise ValueError(f"{name} must be strictly between 0 and 1")
    return v


def int_ge(value: SupportsInt, name: str, min_val: int = 0) -> int:
    """Return ``value`` as ``int`` if >= ``min_val``, else raise ``ValueError``."""
    if not isinstance(value, Integral) or int(value) < min_val:
        raise ValueError(f"{name} must be an integer >= {min_val}")
    return int(value)
