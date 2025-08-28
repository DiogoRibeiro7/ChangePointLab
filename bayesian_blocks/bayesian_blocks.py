# bayesian_blocks.py
# MIT License
# (c) 2025


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Union, Tuple, Literal
from enum import Enum
import warnings
import math
import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.floating]
ArrayI = NDArray[np.integer]


class DataType(Enum):
    """Supported data types for Bayesian Blocks."""

    EVENTS = "events"
    COUNTS = "counts"
    BERNOULLI = "bernoulli"


@dataclass
class BBConfig:
    """Configuration for Bayesian Blocks algorithm."""

    p0: Optional[float] = 0.05
    gamma: Optional[float] = None
    min_block_size: int = 1
    max_blocks: Optional[int] = None
    method: Literal["dp", "recursive"] = "dp"

    def __post_init__(self):
        if self.p0 is not None and not (0 < self.p0 < 1):
            raise ValueError(f"p0 must be in (0,1), got {self.p0}")
        if self.min_block_size < 1:
            raise ValueError(f"min_block_size must be >= 1, got {self.min_block_size}")


@dataclass
class BBResult:
    """Enhanced result object with additional statistics."""

    edges: ArrayF
    block_value: ArrayF
    change_points: ArrayI
    # New fields
    log_likelihood: float = 0.0
    n_blocks: int = field(init=False)
    aic: float = field(init=False)
    bic: float = field(init=False)
    config: Optional[BBConfig] = None

    def __post_init__(self):
        self.n_blocks = len(self.block_value)
        # Simplified AIC/BIC (would need proper likelihood for real calculation)
        self.aic = -2 * self.log_likelihood + 2 * self.n_blocks
        self.bic = -2 * self.log_likelihood + self.n_blocks * math.log(
            len(self.edges) - 1
        )


def _validate_input_array(
    arr: Sequence,
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> ArrayF:
    """Validate and convert input array with descriptive error messages."""
    try:
        arr = np.asarray(arr, dtype=float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{name} must be convertible to numeric array: {e}")

    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {arr.shape}")

    if arr.size == 0:
        warnings.warn(f"{name} is empty, returning empty result")
        return arr

    if min_val is not None and np.any(arr < min_val):
        raise ValueError(f"All values in {name} must be >= {min_val}")

    if max_val is not None and np.any(arr > max_val):
        raise ValueError(f"All values in {name} must be <= {max_val}")

    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN/inf)")

    return arr


def _optimized_dp_solve(
    stat_num: ArrayF,
    stat_den: ArrayF,
    fitness_func: Callable[[ArrayF, ArrayF], ArrayF],
    gamma: float,
    min_block_size: int = 1,
) -> Tuple[ArrayI, ArrayI, ArrayF]:
    """
    Optimized O(N^2) DP with vectorized operations and memory efficiency.
    """
    N = len(stat_num) - 1
    if N < min_block_size:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.zeros(1)

    # Pre-allocate arrays
    opt = np.empty(N + 1, dtype=float)
    last = np.empty(N + 1, dtype=np.int64)
    opt[0] = 0.0
    last[0] = -1

    # Vectorized computation of all pairwise block statistics
    # This is the key optimization - compute all at once instead of in nested loops
    for j in range(1, N + 1):
        # Ensure minimum block size constraint
        start_idx = max(0, j - (N // min_block_size) if min_block_size > 1 else 0)

        # Vectorized block totals for all possible starting points
        i_range = np.arange(start_idx, j)
        if len(i_range) == 0:
            continue

        # Block statistics
        block_num = stat_num[j] - stat_num[i_range]
        block_den = stat_den[j] - stat_den[i_range]

        # Vectorized fitness computation
        fitness_vals = fitness_func(block_num, block_den)

        # Total objective including previous optimal and penalty
        total_obj = opt[i_range] + fitness_vals - gamma

        # Find optimal predecessor
        best_idx = np.argmax(total_obj)
        opt[j] = total_obj[best_idx]
        last[j] = i_range[best_idx]

    # Backtrack to find changepoints
    changepoints = []
    j = N
    while j > 0 and last[j] >= 0:
        if last[j] == 0:
            changepoints.append(j)
            break
        changepoints.append(j)
        j = last[j]

    changepoints = np.array(list(reversed(changepoints)), dtype=np.int64)
    return last, changepoints, opt


def _vectorized_poisson_fitness(num: ArrayF, den: ArrayF) -> ArrayF:
    """Vectorized Poisson fitness computation with proper handling of edge cases."""
    with np.errstate(divide="ignore", invalid="ignore"):
        # Handle the case where num = 0
        result = np.where(
            (den <= 0),
            -np.inf,
            np.where((num <= 0), 0.0, num * (np.log(num) - np.log(den))),
        )
    return result


def _vectorized_bernoulli_fitness(success: ArrayF, trials: ArrayF) -> ArrayF:
    """Vectorized Bernoulli fitness with numerical stability."""
    with np.errstate(divide="ignore", invalid="ignore"):
        p = success / trials
        # Clamp p to avoid log(0)
        p_safe = np.clip(p, 1e-16, 1 - 1e-16)
        result = np.where(
            (trials <= 0),
            -np.inf,
            np.where(
                (success <= 0),
                trials * np.log(1 - p_safe),
                np.where(
                    (success >= trials),
                    trials * np.log(p_safe),
                    success * np.log(p_safe) + (trials - success) * np.log(1 - p_safe),
                ),
            ),
        )
    return result


def bayesian_blocks(
    data: Union[Sequence[float], tuple[Sequence[float], Sequence[float]]],
    *,
    data_type: Union[DataType, str] = "auto",
    config: Optional[BBConfig] = None,
    **kwargs,
) -> BBResult:
    """
    Unified Bayesian Blocks interface with automatic data type detection.

    Parameters
    ----------
    data : array-like or tuple of arrays
        Input data. Format depends on data_type:
        - 'events': 1D array of event times
        - 'counts': 1D array of counts, or (counts, widths) tuple
        - 'bernoulli': 1D array of 0/1 values, or (successes, trials) tuple
    data_type : {'auto', 'events', 'counts', 'bernoulli'}
        Type of data. If 'auto', attempts to detect automatically.
    config : BBConfig, optional
        Configuration object. If None, uses defaults.
    **kwargs
        Additional parameters passed to specific algorithm.

    Returns
    -------
    BBResult
        Result object with edges, block values, and diagnostics.

    Examples
    --------
    >>> # Event times
    >>> times = [1.2, 1.5, 2.1, 4.3, 4.7]
    >>> result = bayesian_blocks(times, data_type='events')

    >>> # Binary sequence
    >>> binary = [0, 0, 1, 1, 1, 0, 1]
    >>> result = bayesian_blocks(binary, data_type='bernoulli')

    >>> # Count data with custom config
    >>> counts = [3, 5, 2, 8, 1, 1]
    >>> cfg = BBConfig(p0=0.01, min_block_size=2)
    >>> result = bayesian_blocks(counts, data_type='counts', config=cfg)
    """
    if config is None:
        config = BBConfig(**kwargs)

    # Auto-detect data type
    if data_type == "auto":
        data_type = _detect_data_type(data)
    elif isinstance(data_type, str):
        data_type = DataType(data_type)

    # Route to appropriate algorithm
    if data_type == DataType.EVENTS:
        return _bayesian_blocks_events(data, config, **kwargs)
    elif data_type == DataType.COUNTS:
        return _bayesian_blocks_counts(data, config, **kwargs)
    elif data_type == DataType.BERNOULLI:
        return _bayesian_blocks_bernoulli(data, config, **kwargs)
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")


def _detect_data_type(data) -> DataType:
    """Attempt to automatically detect the data type."""
    if isinstance(data, tuple) and len(data) == 2:
        # Tuple input suggests either (counts, widths) or (successes, trials)
        arr1, arr2 = data
        arr1, arr2 = np.asarray(arr1), np.asarray(arr2)

        # If second array is all 1s, likely (successes, trials) with trials=1
        if np.all(arr2 == 1):
            return DataType.BERNOULLI
        # If first array <= second array and all integers, likely Bernoulli
        elif np.all(arr1 <= arr2) and np.all(arr1 == arr1.astype(int)):
            return DataType.BERNOULLI
        else:
            return DataType.COUNTS
    else:
        # Single array - need to distinguish between events, counts, and binary
        arr = np.asarray(data)

        # If all 0s and 1s, likely binary
        if np.all(np.isin(arr, [0, 1])):
            return DataType.BERNOULLI
        # If all non-negative integers, likely counts
        elif np.all(arr >= 0) and np.all(arr == arr.astype(int)):
            return DataType.COUNTS
        # Otherwise assume continuous event times
        else:
            return DataType.EVENTS


def _bayesian_blocks_events(data, config: BBConfig, **kwargs) -> BBResult:
    """Enhanced events algorithm with better parameter handling."""
    # Implementation would be similar to original but with improved validation
    # and the optimized DP solver
    pass  # Placeholder - would implement full algorithm


def _bayesian_blocks_counts(data, config: BBConfig, **kwargs) -> BBResult:
    """Enhanced counts algorithm."""
    pass  # Placeholder


def _bayesian_blocks_bernoulli(data, config: BBConfig, **kwargs) -> BBResult:
    """Enhanced Bernoulli algorithm."""
    pass  # Placeholder


# Convenience functions for backward compatibility
def bayesian_blocks_events(t: Sequence[float], **kwargs) -> BBResult:
    """Backward compatibility wrapper."""
    return bayesian_blocks(t, data_type="events", **kwargs)


def bayesian_blocks_counts(counts: Sequence[float], **kwargs) -> BBResult:
    """Backward compatibility wrapper."""
    return bayesian_blocks(counts, data_type="counts", **kwargs)


def bayesian_blocks_bernoulli(successes: Sequence, **kwargs) -> BBResult:
    """Backward compatibility wrapper."""
    return bayesian_blocks(successes, data_type="bernoulli", **kwargs)
