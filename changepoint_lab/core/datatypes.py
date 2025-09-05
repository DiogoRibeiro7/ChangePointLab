from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class ChangePointResult:
    """Container for detected change points.

    Parameters
    ----------
    indices : np.ndarray
        Sorted 1D array of changepoint indices.
    score : float, optional
        Optional overall score or objective value.
    metadata : Mapping[str, Any], optional
        Additional algorithm-specific outputs.
    """

    indices: np.ndarray
    score: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
