# data_loader.py
# MIT License
# (c) 2025

from __future__ import annotations

import datetime as dt
from typing import Tuple, Optional, List

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
) -> Tuple[np.ndarray, int]:
    """
    Load event data from a CSV file and convert to binary time series.
    
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
        Hour of day to align as the first bin (0-23).
    days_span : Optional[int]
        If provided, force the data to span exactly this many days from the 
        earliest timestamp. Otherwise, use the natural span of the data.
        
    Returns
    -------
    Tuple[np.ndarray, int]
        binary_series: Boolean array where True indicates bin with event(s)
        bins_per_day: Number of bins per day (24*60/bin_minutes)
    """
    # Validate parameters
    if bin_minutes <= 0 or bin_minutes > 1440 or 1440 % bin_minutes != 0:
        raise ValueError(f"bin_minutes must be a positive divisor of 1440 (minutes in a day), got {bin_minutes}")
    
    if not (0 <= start_hour < 24):
        raise ValueError(f"start_hour must be in range [0, 23], got {start_hour}")
    
    # Load data
    df = pd.read_csv(filepath)
    
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in CSV")
    
    if value_col is not None and value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in CSV")
    
    # Convert timestamps to datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Filter events based on value threshold if applicable
    if value_col is not None:
        df = df[df[value_col] > value_threshold]
    
    if df.empty:
        return np.array([], dtype=bool), 24 * 60 // bin_minutes
    
    # Sort by timestamp
    df = df.sort_values(timestamp_col)
    
    # Calculate bins per day
    bins_per_day = 24 * 60 // bin_minutes
    minutes_per_bin = bin_minutes
    
    # Get time range
    min_time = df[timestamp_col].min()
    
    # Adjust min_time to start_hour
    min_date = min_time.normalize()  # Get date at midnight
    min_time_adjusted = min_date + dt.timedelta(hours=start_hour)
    
    if min_time < min_time_adjusted:
        min_time_adjusted -= dt.timedelta(days=1)
    
    # Calculate end time
    if days_span is not None:
        max_time = min_time_adjusted + dt.timedelta(days=days_span)
    else:
        max_time = df[timestamp_col].max() + dt.timedelta(minutes=minutes_per_bin)
    
    # Create time bins
    time_bins = pd.date_range(start=min_time_adjusted, end=max_time, freq=f"{minutes_per_bin}min")
    
    # Bin events
    binned = pd.cut(df[timestamp_col], bins=time_bins, right=False, labels=False)
    
    # Create binary series
    total_bins = len(time_bins) - 1
    binary_series = np.zeros(total_bins, dtype=bool)
    
    # Mark bins with events as True
    for bin_idx in binned.dropna():
        if 0 <= bin_idx < total_bins:  # Guard against out-of-bounds
            binary_series[int(bin_idx)] = True
    
    return binary_series, bins_per_day


def parse_binary_string(binary_str: str) -> np.ndarray:
    """
    Parse a string of 0s and 1s into a binary array.
    Useful for testing with manually specified patterns.
    
    Parameters
    ----------
    binary_str : str
        String containing only 0s and 1s (whitespace is ignored).
        
    Returns
    -------
    np.ndarray
        Boolean array where True corresponds to '1' in the input.
    """
    # Remove whitespace
    clean_str = ''.join(binary_str.split())
    
    if not all(c in '01' for c in clean_str):
        raise ValueError("Input must contain only 0s and 1s")
    
    return np.array([c == '1' for c in clean_str], dtype=bool)


if __name__ == "__main__":
    # Simple test
    import tempfile
    
    # Create a temporary CSV with timestamps
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
        f.write("timestamp,value\n")
        base_time = dt.datetime(2025, 1, 1, 0, 0)
        for i in range(100):
            # Add events every hour, with some random noise
            event_time = base_time + dt.timedelta(minutes=60*i + np.random.randint(-10, 10))
            value = np.random.rand()
            f.write(f"{event_time.isoformat()},{value}\n")
        temp_filename = f.name
    
    # Test loading
    try:
        binary_data, bins_per_day = load_binary_from_csv(
            temp_filename, 
            timestamp_col="timestamp", 
            value_col="value", 
            value_threshold=0.5,
            bin_minutes=15
        )
        print(f"Loaded {binary_data.sum()} events in {len(binary_data)} bins")
        print(f"Bins per day: {bins_per_day}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        import os
        os.unlink(temp_filename)
