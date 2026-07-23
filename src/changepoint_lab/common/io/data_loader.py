# data_loader.py
# MIT License
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def load_binary_from_csv(
    csv_path: str | Path,
    *,
    timestamp_col: str = "timestamp",
    value_col: str | None = None,
    value_threshold: float = 0.0,
    bin_minutes: int = 15,
    start_hour: int = 0,
    days_span: Optional[int] = None,
    timezone: Optional[str] = None,
    return_time_bins: bool = False,
) -> Union[Tuple[NDArray[np.bool_], int], Tuple[NDArray[np.bool_], int, pd.DatetimeIndex]]:
    """Load event data from CSV and convert to a binned boolean series.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.
    timestamp_col : str
        Name of the timestamp column.
    value_col : str, optional
        If provided, threshold this column to determine events.
    value_threshold : float
        Threshold for event detection when value_col is provided.
    bin_minutes : int
        Size of each time bin in minutes.
    start_hour : int
        Local hour of day to align as the first bin (0-23).
    days_span : int, optional
        If provided, force the data to span exactly this many days from the
        earliest timestamp (using LOCAL calendar days in the chosen timezone).
    timezone : str, optional
        IANA timezone for local "wall clock" binning (e.g., "Europe/Lisbon").
        - If timestamps are naive, tz-localize with nonexistent='shift_forward'
          and ambiguous='NaT' (dropping NaT rows).
        - If timestamps are tz-aware, convert into `timezone`.
        - If None, leave times naive.
    return_time_bins : bool
        If True, also return the DatetimeIndex of bin edges (left-closed).

    Returns
    -------
    binary_series : np.ndarray[bool]
        True if any event falls into the bin.
    bins_per_day : int
        24 * 60 / bin_minutes (number of bins in a standard 24h day).
    time_bins : pd.DatetimeIndex, optional
        Bin edges as a tz-aware (if timezone given) or naive index.
    """
    if bin_minutes <= 0 or bin_minutes > 1440 or 1440 % bin_minutes != 0:
        raise ValueError(
            f"bin_minutes must be a positive divisor of 1440, got {bin_minutes}"
        )
    if not (0 <= start_hour < 24):
        raise ValueError(f"start_hour must be in range [0, 23], got {start_hour}")

    df = pd.read_csv(csv_path)
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in CSV")
    if value_col is not None and value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in CSV")

    ts = pd.to_datetime(df[timestamp_col], errors="coerce", utc=False)

    if timezone:
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(
                timezone,
                nonexistent="shift_forward",
                ambiguous="NaT",
            )
        else:
            ts = ts.dt.tz_convert(timezone)
        mask_valid = ~ts.isna()
        if not mask_valid.all():
            df = df.loc[mask_valid].copy()
            ts = ts.loc[mask_valid]
    df[timestamp_col] = ts

    if value_col is not None:
        df = df[df[value_col] > value_threshold]
    if df.empty:
        out = (np.array([], dtype=bool), 24 * 60 // bin_minutes)
        return (*out, pd.DatetimeIndex([], dtype="datetime64[ns]")) if return_time_bins else out

    df = df.sort_values(timestamp_col)
    bins_per_day = 24 * 60 // bin_minutes
    minutes_per_bin = int(bin_minutes)

    min_time = df[timestamp_col].min()
    if timezone:
        start_anchor = min_time.normalize() + pd.Timedelta(hours=start_hour)
        if min_time < start_anchor:
            min_time_adjusted = start_anchor
        else:
            min_time_adjusted = start_anchor + pd.Timedelta(days=1)
    else:
        min_date = min_time.normalize()
        min_time_adjusted = min_date + dt.timedelta(hours=start_hour)
        if min_time >= min_time_adjusted:
            min_time_adjusted = min_time_adjusted + dt.timedelta(days=1)

    max_time = df[timestamp_col].max()
    if days_span is not None and days_span > 0:
        max_time_inc = min_time_adjusted + pd.Timedelta(days=int(days_span))
    else:
        max_time_inc = max_time + pd.Timedelta(minutes=minutes_per_bin)

    time_bins = pd.date_range(
        start=min_time_adjusted,
        end=max_time_inc,
        freq=f"{minutes_per_bin}min",
        inclusive="both",
    )

    binned = pd.cut(
        df[timestamp_col],
        bins=time_bins.to_list(),
        right=False,
        labels=False,
        include_lowest=True,
    )

    total_bins = max(len(time_bins) - 1, 0)
    binary_series = np.zeros(total_bins, dtype=bool)
    binned = binned.dropna().astype(int)
    if not binned.empty:
        idx = binned.to_numpy()
        idx = idx[(0 <= idx) & (idx < total_bins)]
        if idx.size:
            binary_series[np.unique(idx)] = True

    if return_time_bins:
        return binary_series, bins_per_day, time_bins
    return binary_series, bins_per_day


def empirical_per_bin_mean(x: NDArray[np.bool_], N: int) -> NDArray[np.floating]:
    """Compute per-bin mean across days for a binary series of shape (D*N,)."""
    if x.ndim != 1 or x.size % N != 0:
        raise ValueError("x must be 1-D with length multiple of N.")
    D = x.size // N
    mat = x.reshape(D, N).astype(float)
    return mat.mean(axis=0)


def parse_binary_string(binary_str: str) -> NDArray[np.bool_]:
    """Parse a string of 0s and 1s into a boolean array."""
    clean = "".join(ch for ch in binary_str if ch in "01")
    if not clean:
        return np.array([], dtype=bool)
    if set(clean) - {"0", "1"}:
        raise ValueError("Input must contain only 0s and 1s")
    return np.frombuffer(clean.encode(), dtype="S1") == b"1"
