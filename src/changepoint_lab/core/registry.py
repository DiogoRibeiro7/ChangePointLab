"""Explicit factories for reproducible offline detector comparisons."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .comparison import ComparableDetector, DetectorComparison, compare_detectors

DetectorFactory: TypeAlias = Callable[[], ComparableDetector]


class DetectorRegistry:
    """Named detector factories with deterministic ordering and no global state.

    Registrations are explicit and local to each registry. Factories are called
    afresh when ``build`` or ``compare`` is invoked. To obtain independent
    repetitions, factories must construct new detector instances.
    """

    def __init__(self) -> None:
        self._factories: dict[str, DetectorFactory] = {}

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered names in registration order."""
        return tuple(self._factories)

    def register(self, name: str, factory: DetectorFactory) -> None:
        """Register one zero-argument factory; reject duplicate names."""
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError("name must be a nonempty, unpadded string")
        if not callable(factory):
            raise TypeError("factory must be callable")
        if name in self._factories:
            raise ValueError(f"detector {name!r} is already registered")
        self._factories[name] = factory

    def build(self, names: Sequence[str] | None = None) -> dict[str, ComparableDetector]:
        """Instantiate the selected factories, preserving requested order.

        ``names=None`` selects all registered detectors. Missing or repeated
        selections are rejected rather than silently discarded.
        """
        selected = self.names if names is None else tuple(names)
        if len(selected) != len(set(selected)):
            raise ValueError("selected detector names must be unique")
        if not selected:
            raise ValueError("no detectors selected")
        built: dict[str, ComparableDetector] = {}
        for name in selected:
            if name not in self._factories:
                raise KeyError(f"unknown detector {name!r}")
            try:
                detector = self._factories[name]()
            except Exception as exc:
                raise RuntimeError(f"factory for detector {name!r} failed") from exc
            if not callable(getattr(detector, "fit_predict", None)):
                raise TypeError(f"factory for detector {name!r} returned an incompatible object")
            built[name] = detector
        return built

    def compare(
        self,
        data: NDArray[np.floating] | Sequence[float],
        *,
        names: Sequence[str] | None = None,
        actual: Sequence[int] | NDArray[np.integer] | None = None,
        tolerance: int = 0,
    ) -> tuple[DetectorComparison, ...]:
        """Compare newly built offline detectors with common reference data."""
        return compare_detectors(self.build(names), data, actual=actual, tolerance=tolerance)
