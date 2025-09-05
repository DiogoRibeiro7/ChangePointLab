# data_loader.py
# MIT License
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


def _parse_iso(ts: str) -> dt.datetime:
    """
    Parse common ISO-8601 timestamps (e.g., '2025-08-28T09:15:00' or with 'Z').
    Naive datetimes treated as UTC by default (no tz conversion is applied).
    """
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1]
    # Support fractional seconds
    fmts = ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")
    for f in fmts:
        try:
            return dt.datetime.strptime(ts, f)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized timestamp format: {ts!r}")


def load_binary_from_csv(
    csv_path: str | Path,
    *,
    timestamp_col: str = "timestamp",
    value_col: str | None = None,
    value_threshold: float = 0.0,
    bin_minutes: int = 15,
    start_hour: int = 0,
    days_span: Optional[int] = None,
) -> Tuple[NDArray[np.bool_], int]:
    """
    Convert a CSV of timestamps (and optional value) to a binary time series at fixed bins per day.

    Semantics
    ---------
    - Each row marks an "event". If 'value_col' is None, every row counts as an event.
      Otherwise, rows with value > value_threshold count as events.
    - For each calendar day (based on local naive datetime), we split the 24h day into
      N = 24 * 60 / bin_minutes bins, anchored so that index 0 corresponds to start_hour:00.
    - If at least one event falls inside a bin, that bin is marked True for that day (else False).
    - Output x is a 1-D boolean array of length (#days * N), day-major order.

    Parameters
    ----------
    csv_path : path to a CSV with at least 'timestamp_col'
    timestamp_col : column name containing timestamps (ISO-8601 compatible)
    value_col : optional numeric column; if provided, filter events with value > value_threshold
    value_threshold : threshold for value_col
    bin_minutes : number of minutes per bin (e.g., 15 => N=96)
    start_hour : hour-of-day that maps to index 0 (0..23)
    days_span : optionally force an exact number of consecutive days starting from min date

    Returns
    -------
    x : np.ndarray[bool] shape (D*N,)
    N : int (bins per day)
    """
    path = Path(csv_path)
    N = int(24 * 60 // bin_minutes)
    bins_per_hour = N // 24
    assert (24 * 60) % bin_minutes == 0, "bin_minutes must divide 1440."

    # Collect parsed timestamps (and optional values)
    stamps: List[dt.datetime] = []
    vals: List[float] = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if timestamp_col not in reader.fieldnames:
            raise ValueError(f"CSV must have column {timestamp_col!r}")
        for row in reader:
            ts = _parse_iso(row[timestamp_col])
            if value_col is not None:
                try:
                    v = float(row[value_col])
                except Exception:
                    continue
                if not (v > value_threshold):
                    continue
            stamps.append(ts)

    if not stamps:
        return np.array([], dtype=bool), N

    # Build day indices
    dates = [s.date() for s in stamps]
    day0 = min(dates)
    if days_span is None:
        day_last = max(dates)
        D = (day_last - day0).days + 1
    else:
        D = int(days_span)

    def day_index(d: dt.date) -> int:
        return (d - day0).days

    # Populate (day, bin) occupancy
    x = np.zeros(D * N, dtype=bool)
    for ts in stamps:
        d = day_index(ts.date())
        if not (0 <= d < D):
            # outside forced span
            continue
        # seconds since start_hour
        sec = (ts.hour - start_hour) * 3600 + ts.minute * 60 + ts.second
        sec %= 24 * 3600
        bin_idx = int(sec // (bin_minutes * 60))
        x[d * N + bin_idx] = True

    return x, N


def empirical_per_bin_mean(x: NDArray[np.bool_], N: int) -> NDArray[np.floating]:
    """
    Compute per-bin mean across days for a binary series of shape (D*N,).
    """
    if x.ndim != 1 or x.size % N != 0:
        raise ValueError("x must be 1-D with length multiple of N.")
    D = x.size // N
    mat = x.reshape(D, N).astype(float)
    return mat.mean(axis=0)


def parse_binary_string(binary_str: str) -> NDArray[np.bool_]:
    """Parse a string of 0s and 1s into a boolean array.

    Whitespace characters are ignored. Any character other than ``0`` or ``1``
    raises a :class:`ValueError`.
    """
    clean = "".join(ch for ch in binary_str if ch in "01")
    if not clean:
        return np.array([], dtype=bool)
    if set(clean) - {"0", "1"}:
        raise ValueError("Input must contain only 0s and 1s")
    return np.frombuffer(clean.encode(), dtype="S1") == b"1"


# from data_loader import load_binary_from_csv, empirical_per_bin_mean
# x, N = load_binary_from_csv("events.csv", timestamp_col="ts", bin_minutes=15, start_hour=0)
# # Fit model with prior.N == N
