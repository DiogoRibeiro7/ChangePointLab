from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..core.datatypes import ChangePointResult
from ..core.validation import require_ndarray, validate_array_shape

ArrayLike = np.ndarray


class BaseDetector(ABC):
    """Common interface all detectors must implement."""

    @abstractmethod
    def fit(self, x: ArrayLike) -> BaseDetector:
        """Fit internal state to `x` (if needed)."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: ArrayLike | None = None) -> ChangePointResult:
        """Detect change points on `x` or the data seen in ``fit``."""
        raise NotImplementedError

    def fit_predict(self, x: ArrayLike) -> ChangePointResult:
        """Fit the model and immediately predict on the same data."""
        self._validate_input(x)
        return self.fit(x).predict()

    def get_params(self) -> dict[str, Any]:
        """Return public constructor parameters for estimator-style wrappers."""
        return {
            name: value
            for name, value in vars(self).items()
            if not name.startswith("_")
        }

    def _validate_input(self, x: ArrayLike) -> None:
        validate_array_shape(require_ndarray(x), name="x", ndim=(1, 2), non_empty=True)
