from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..core.datatypes import ChangePointResult

ArrayLike = np.ndarray


class BaseDetector(ABC):
    """Common interface all detectors must implement."""

    @abstractmethod
    def fit(self, x: ArrayLike) -> "BaseDetector":
        """Fit internal state to `x` (if needed)."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: Optional[ArrayLike] = None) -> ChangePointResult:
        """Detect change points on `x` or the data seen in ``fit``."""
        raise NotImplementedError

    def fit_predict(self, x: ArrayLike) -> ChangePointResult:
        """Fit the model and immediately predict on the same data."""
        self._validate_input(x)
        return self.fit(x).predict()

    # minimal input validation
    def _validate_input(self, x: ArrayLike) -> None:
        if not isinstance(x, np.ndarray):
            raise TypeError(f"`x` must be np.ndarray, got {type(x)!r}")
        if x.ndim not in (1, 2):
            raise ValueError(f"`x` must be 1D or 2D, got x.ndim = {x.ndim}")
        if x.size == 0:
            raise ValueError("`x` cannot be empty.")
