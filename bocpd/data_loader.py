# data_loader.py
# MIT License
# (c) 2025

from __future__ import annotations

import datetime as dt
from typing import Tuple, Optional, List, Union

import numpy as np
import pandas as pd


def load_binary_from_csv(
    filepath: str,
    timestamp_col: str = "timestamp",
    value_col: Optional[str] = None,
    value_threshold: float = 0.0,
    bin_minutes: int = 15,
    start_hour: int = 0,
    days_span: Optional[int] = None,
    timezone: Optional[str] = None,
    return_time_bins: bool = False,
) -> Union[Tuple[np.ndarray, int], Tuple[np.ndarray, int, pd.DatetimeIndex]]:
    """
    Load event data from a CSV file and convert to a binned boolean series.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    timestamp_col : str
        Name of the timestamp column.
    value_col : Optional[str]
        If provided, use this column to determine events by thresholding.
        If None, each row is treated as an event.
    value_threshold : float
        Threshold value for event detection when value_col is provided.
    bin_minutes : int
        Size of each time bin in minutes.
    start_hour : int
        Local hour of day to align as the first bin (0-23).
    days_span : Optional[int]
        If provided, force the data to span exactly this many days from the
        earliest timestamp (using LOCAL calendar days in the chosen timezone).
    timezone : Optional[str]
        IANA timezone for local “wall clock” binning (e.g., "Europe/Lisbon").
        - If timestamps are naive, we tz-localize with
          nonexistent='shift_forward', ambiguous='NaT' (dropping 'NaT' rows).
        - If timestamps are already tz-aware, we tz-convert into `timezone`.
        - If None, we keep naive times (DST behavior is then undefined).
    return_time_bins : bool
        If True, also return the DatetimeIndex of bin edges (left-closed).

    Returns
    -------
    (binary_series, bins_per_day) or (binary_series, bins_per_day, time_bins)
        binary_series : np.ndarray[bool]
            True if any event falls into the bin.
        bins_per_day : int
            24*60 / bin_minutes (number of bins in a standard 24h day).
        time_bins : pd.DatetimeIndex
            Bin edges as a tz-aware (if timezone given) or naive index.
            Left-closed / right-open intervals: [edge_i, edge_{i+1})
    """
    # Validate parameters
    if bin_minutes <= 0 or bin_minutes > 1440 or 1440 % bin_minutes != 0:
        raise ValueError(f"bin_minutes must be a positive divisor of 1440, got {bin_minutes}")

    if not (0 <= start_hour < 24):
        raise ValueError(f"start_hour must be in range [0, 23], got {start_hour}")

    # Load
    df = pd.read_csv(filepath)
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in CSV")

    if value_col is not None and value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in CSV")

    # Parse timestamps
    ts = pd.to_datetime(df[timestamp_col], errors="coerce", utc=False)

    if timezone:
        # If tz-naive -> localize; if tz-aware -> convert
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(
                timezone,
                nonexistent="shift_forward",  # spring-forward gap -> shift forward
                ambiguous="NaT",              # fall-back overlap -> mark as NaT
            )
        else:
            ts = ts.dt.tz_convert(timezone)

        # Drop invalid timestamps created by DST disambiguation
        mask_valid = ~ts.isna()
        if not mask_valid.all():
            df = df.loc[mask_valid].copy()
            ts = ts.loc[mask_valid]
    else:
        # Leave as naive; DST edges will be treated as plain arithmetic.
        pass

    df[timestamp_col] = ts

    # Filter events by value threshold
    if value_col is not None:
        df = df[df[value_col] > value_threshold]

    if df.empty:
        out = (np.array([], dtype=bool), 24 * 60 // bin_minutes)
        return (*out, pd.DatetimeIndex([], dtype="datetime64[ns]")) if return_time_bins else out

    # Sort by timestamp
    df = df.sort_values(timestamp_col)

    # Bins per "standard" day (24h)
    bins_per_day = 24 * 60 // bin_minutes
    minutes_per_bin = int(bin_minutes)

    # Determine the local-anchored start (midnight + start_hour) in the chosen tz
    min_time = df[timestamp_col].min()
    if timezone:
        # local midnight
        start_anchor = min_time.normalize() + pd.Timedelta(hours=start_hour)
        # If min_time < anchor (same date), keep; else start at next day's anchor
        if min_time < start_anchor:
            min_time_adjusted = start_anchor
        else:
            min_time_adjusted = (start_anchor + pd.Timedelta(days=1))
    else:
        # naive path (original behavior)
        min_date = min_time.normalize()
        min_time_adjusted = min_date + dt.timedelta(hours=start_hour)
        if min_time >= min_time_adjusted:
            min_time_adjusted = min_time_adjusted + dt.timedelta(days=1)

    # Compute end bound
    max_time = df[timestamp_col].max()
    if days_span is not None and days_span > 0:
        # Cover exactly `days_span` LOCAL days from the adjusted start
        max_time_inc = min_time_adjusted + pd.Timedelta(days=int(days_span))
    else:
        # Cover through last observed event + one bin
        max_time_inc = max_time + pd.Timedelta(minutes=minutes_per_bin)

    # Build bin edges as a DatetimeIndex (left-closed)
    time_bins = pd.date_range(
        start=min_time_adjusted,
        end=max_time_inc,
        freq=f"{minutes_per_bin}min",
        inclusive="both"
    )

    # Assign each event to a bin (left-closed, right-open)
    binned = pd.cut(
        df[timestamp_col],
        bins=time_bins,
        right=False,
        labels=False,
        include_lowest=True,  # include the very first boundary cleanly
    )

    # Create binary series
    total_bins = max(len(time_bins) - 1, 0)
    binary_series = np.zeros(total_bins, dtype=bool)

    # Mark bins containing events as True
    binned = binned.dropna().astype(int)
    if not binned.empty:
        idx = binned.to_numpy()
        idx = idx[(0 <= idx) & (idx < total_bins)]
        if idx.size:
            binary_series[np.unique(idx)] = True

    if return_time_bins:
        return binary_series, bins_per_day, time_bins
    else:
        return binary_series, bins_per_day


def parse_binary_string(binary_str: str) -> np.ndarray:
    """
    Parse a string of 0s and 1s into a binary array. Useful for tests.

    Parameters
    ----------
    binary_str : str
        String containing only 0s and 1s (whitespace is ignored).
    """
    clean_str = "".join(ch for ch in binary_str if ch in "01")
    if not clean_str:
        return np.array([], dtype=bool)
    if set(clean_str) - set("01"):
        raise ValueError("Input must contain only 0s and 1s")
    return np.frombuffer(clean_str.encode(), dtype="S1") == b"1"
