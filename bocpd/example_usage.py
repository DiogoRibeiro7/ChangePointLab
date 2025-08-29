# example_usage.py
# MIT License
# (c) 2025

"""
Example usage of the BOCPD (Bayesian Online Changepoint Detection) API.
This script demonstrates different ways to use the BOCPD implementation
for both offline (batch) and online (streaming) scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd

from bocpd.bocpd import (
    BOCPD,
    BOCPDConfig,
    ConstantHazard,
    ScheduledHazard,
    BoostedBoundaryHazard,
)
from bocpd.bocpd_plotting import plot_run_length_heatmap, plot_cp_probability
from common.io.data_loader import load_binary_from_csv, parse_binary_string


def example_1_basic_usage():
    """Basic usage with synthetic data - offline processing."""
    print("\n=== Example 1: Basic Usage ===")
    
    # Create synthetic data with a changepoint
    np.random.seed(42)
    x1 = np.random.binomial(1, 0.1, size=50)  # Low probability
    x2 = np.random.binomial(1, 0.8, size=50)  # High probability
    x = np.concatenate([x1, x2])
    
    # Create BOCPD model with constant hazard
    hazard = ConstantHazard(mean_run_length=50.0)
    config = BOCPDConfig(alpha0=1.0, beta0=1.0, max_run_length=100)
    model = BOCPD(hazard, config)
    
    # Process the data in batch mode
    result = model.run(x)
    
    # Print summary
    print(f"Data length: {len(x)}")
    print(f"True changepoint at t=50")
    
    # Find detected changepoints (where probability exceeds threshold)
    threshold = 0.5
    cp_indices = np.where(result.cp_prob > threshold)[0]
    print(f"Detected changepoints (p > {threshold}): {cp_indices}")
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 3]})
    
    # Plot the binary data
    axes[0].step(np.arange(len(x)), x, where='post', label='Data')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Binary Sequence')
    axes[0].axvline(x=50, color='r', linestyle='--', alpha=0.5, label='True CP')
    axes[0].legend()
    
    # Plot the changepoint probability
    plot_cp_probability(result.cp_prob, ax=axes[1])
    axes[1].axhline(y=threshold, color='g', linestyle='--', label=f'Threshold ({threshold})')
    axes[1].axvline(x=50, color='r', linestyle='--', alpha=0.5, label='True CP')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("example1_basic_usage.png")
    plt.close()
    
    print(f"Plot saved to example1_basic_usage.png")


def example_2_online_processing():
    """Demonstrate online processing with streaming data."""
    print("\n=== Example 2: Online Processing ===")
    
    # Create synthetic data with multiple changepoints
    np.random.seed(123)
    segments = [
        np.random.binomial(1, 0.1, size=40),  # Low probability
        np.random.binomial(1, 0.7, size=30),  # High probability
        np.random.binomial(1, 0.4, size=50),  # Medium probability
    ]
    x = np.concatenate(segments)
    true_cps = [40, 70]
    
    # Create BOCPD model
    hazard = ConstantHazard(mean_run_length=40.0)
    model = BOCPD(hazard)
    
    # Process data online (one point at a time)
    cp_probs = []
    map_rls = []
    pred_means = []
    
    for t, x_t in enumerate(x):
        # Process point
        result = model.update(x_t)
        
        # Store results
        cp_probs.append(result["cp_prob"])
        map_rls.append(result["map_run_length"])
        pred_means.append(result["pred_mean"])
        
        # Print result when we detect a changepoint
        if result["cp_prob"] > 0.5:
            print(f"Detected changepoint at t={t} with probability {result['cp_prob']:.4f}")
    
    # Compare with batch processing
    batch_result = BOCPD(hazard).run(x)
    
    # Verify online and batch are equivalent
    online_cp_probs = np.array(cp_probs)
    batch_cp_probs = batch_result.cp_prob
    
    if np.allclose(online_cp_probs, batch_cp_probs):
        print("✓ Online and batch processing produce identical results")
    else:
        print("✗ Online and batch processing results differ")
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    # Plot the binary data
    axes[0].step(np.arange(len(x)), x, where='post')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Binary Sequence with True Changepoints')
    for cp in true_cps:
        axes[0].axvline(x=cp, color='r', linestyle='--', alpha=0.5)
    
    # Plot the changepoint probability
    axes[1].plot(np.arange(len(x)), cp_probs)
    axes[1].set_ylabel('CP Probability')
    axes[1].set_title('Changepoint Probability P(r_t=0 | x_{1:t})')
    axes[1].axhline(y=0.5, color='g', linestyle='--')
    for cp in true_cps:
        axes[1].axvline(x=cp, color='r', linestyle='--', alpha=0.5)
    
    # Plot the run length
    axes[2].plot(np.arange(len(x)), map_rls)
    axes[2].set_xlabel('Time t')
    axes[2].set_ylabel('MAP Run Length')
    axes[2].set_title('Maximum A Posteriori Run Length')
    for cp in true_cps:
        axes[2].axvline(x=cp, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("example2_online_processing.png")
    plt.close()
    
    print(f"Plot saved to example2_online_processing.png")


def example_3_custom_hazard():
    """Demonstrate a custom hazard function for time-of-day patterns."""
    print("\n=== Example 3: Custom Hazard Functions ===")
    
    # Create synthetic daily data (96 points per day = 15 min intervals)
    np.random.seed(456)
    points_per_day = 96
    days = 5
    
    # Create a pattern: different probabilities for different times of day
    morning = np.random.binomial(1, 0.1, size=(days, points_per_day // 4))
    midday = np.random.binomial(1, 0.6, size=(days, points_per_day // 4))
    afternoon = np.random.binomial(1, 0.3, size=(days, points_per_day // 4))
    night = np.random.binomial(1, 0.05, size=(days, points_per_day // 4))
    
    # Combine into daily patterns
    daily_patterns = np.column_stack([morning, midday, afternoon, night])
    x = daily_patterns.flatten()
    
    # Create timestamp series for plotting (15-minute intervals)
    start_time = datetime(2025, 1, 1)
    timestamps = [start_time + timedelta(minutes=15*i) for i in range(len(x))]
    
    # ---- Model with constant hazard (baseline) ----
    constant_hazard = ConstantHazard(mean_run_length=points_per_day * 2)
    constant_model = BOCPD(constant_hazard)
    constant_result = constant_model.run(x)
    
    # ---- Model with scheduled hazard ----
    # Higher hazard at midnight (boundary between days)
    schedule = [0.1 if i % points_per_day == 0 else 0.01 for i in range(points_per_day)]
    scheduled_hazard = ScheduledHazard(schedule=schedule, period=points_per_day)
    scheduled_model = BOCPD(scheduled_hazard)
    scheduled_result = scheduled_model.run(x)
    
    # ---- Model with boundary-boosted hazard ----
    # Base hazard with boost at midnight
    base_hazard = ConstantHazard(mean_run_length=points_per_day * 2)
    boosted_hazard = BoostedBoundaryHazard(
        base=base_hazard, 
        period=points_per_day, 
        boundary_indices=frozenset([0]),  # Boost at t % period == 0
        boost_factor=10.0
    )
    boosted_model = BOCPD(boosted_hazard)
    boosted_result = boosted_model.run(x)
    
    # True daily boundaries (every 96 points)
    true_boundaries = np.arange(points_per_day, len(x), points_per_day)
    
    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Plot data with day boundaries
    axes[0].step(timestamps, x, where='post')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Synthetic Daily Activity Data (15-min intervals)')
    for b in true_boundaries:
        axes[0].axvline(x=timestamps[b], color='r', linestyle='--', alpha=0.5)
    axes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M'))
    
    # Plot constant hazard results
    axes[1].plot(timestamps, constant_result.cp_prob)
    axes[1].set_ylabel('CP Probability')
    axes[1].set_title('Constant Hazard CP Probability')
    for b in true_boundaries:
        axes[1].axvline(x=timestamps[b], color='r', linestyle='--', alpha=0.5)
    axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M'))
    
    # Plot scheduled hazard results
    axes[2].plot(timestamps, scheduled_result.cp_prob)
    axes[2].set_ylabel('CP Probability')
    axes[2].set_title('Scheduled Hazard CP Probability')
    for b in true_boundaries:
        axes[2].axvline(x=timestamps[b], color='r', linestyle='--', alpha=0.5)
    axes[2].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M'))
    
    # Plot boosted hazard results
    axes[3].plot(timestamps, boosted_result.cp_prob)
    axes[3].set_ylabel('CP Probability')
    axes[3].set_title('Boundary-Boosted Hazard CP Probability')
    for b in true_boundaries:
        axes[3].axvline(x=timestamps[b], color='r', linestyle='--', alpha=0.5)
    axes[3].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M'))
    
    plt.tight_layout()
    plt.savefig("example3_custom_hazards.png")
    plt.close()
    
    print(f"Plot saved to example3_custom_hazards.png")
    
    # Compare detection accuracy
    print("\nComparison of hazard functions for day boundary detection:")
    
    def count_correct_detections(cp_probs, true_cps, window=5, threshold=0.5):
        """Count correct changepoint detections within a window."""
        detected = np.where(cp_probs > threshold)[0]
        correct = 0
        
        for true_cp in true_cps:
            # Check if any detection is within the window of the true CP
            if any(abs(d - true_cp) <= window for d in detected):
                correct += 1
        
        return {
            "true_positives": correct,
            "false_positives": len(detected) - correct,
            "false_negatives": len(true_cps) - correct,
            "precision": correct / max(1, len(detected)),
            "recall": correct / len(true_cps)
        }
    
    # Evaluate each model
    constant_metrics = count_correct_detections(constant_result.cp_prob, true_boundaries)
    scheduled_metrics = count_correct_detections(scheduled_result.cp_prob, true_boundaries)
    boosted_metrics = count_correct_detections(boosted_result.cp_prob, true_boundaries)
    
    # Print metrics
    print(f"Constant Hazard: Precision={constant_metrics['precision']:.2f}, Recall={constant_metrics['recall']:.2f}")
    print(f"Scheduled Hazard: Precision={scheduled_metrics['precision']:.2f}, Recall={scheduled_metrics['recall']:.2f}")
    print(f"Boosted Hazard: Precision={boosted_metrics['precision']:.2f}, Recall={boosted_metrics['recall']:.2f}")


def example_4_real_data():
    """Demonstrate using BOCPD with simulated 'real' data from CSV."""
    print("\n=== Example 4: Working with CSV Data ===")
    
    # Create a simulated CSV file with timestamped events
    csv_filename = "simulated_events.csv"
    
    # Generate synthetic timestamps (hourly events for 5 days with pattern changes)
    start_date = datetime(2025, 1, 1)
    timestamps = []
    values = []
    
    # Morning pattern (high frequency 8am-12pm)
    for day in range(5):
        for hour in range(8, 12):
            # Generate 3-5 events per hour
            for _ in range(np.random.randint(3, 6)):
                minute = np.random.randint(0, 60)
                timestamps.append(start_date + timedelta(days=day, hours=hour, minutes=minute))
                values.append(np.random.uniform(0.5, 1.0))  # High values
    
    # Afternoon pattern (medium frequency 1pm-5pm)
    for day in range(5):
        for hour in range(13, 17):
            # Generate 1-3 events per hour
            for _ in range(np.random.randint(1, 4)):
                minute = np.random.randint(0, 60)
                timestamps.append(start_date + timedelta(days=day, hours=hour, minutes=minute))
                values.append(np.random.uniform(0.3, 0.8))  # Medium values
    
    # Evening pattern (low frequency 6pm-10pm)
    for day in range(5):
        for hour in range(18, 22):
            # Generate 0-1 events per hour
            for _ in range(np.random.randint(0, 2)):
                minute = np.random.randint(0, 60)
                timestamps.append(start_date + timedelta(days=day, hours=hour, minutes=minute))
                values.append(np.random.uniform(0.1, 0.5))  # Low values
    
    # Night pattern (very low frequency 11pm-7am)
    for day in range(5):
        for hour in list(range(23, 24)) + list(range(0, 8)):
            # Generate 0-1 events per hour, but with lower probability
            if np.random.random() < 0.3:
                minute = np.random.randint(0, 60)
                timestamps.append(start_date + timedelta(days=day, hours=hour, minutes=minute))
                values.append(np.random.uniform(0.0, 0.3))  # Very low values
    
    # Create DataFrame and sort by timestamp
    df = pd.DataFrame({
        "timestamp": timestamps,
        "value": values
    })
    df = df.sort_values("timestamp")
    
    # Save to CSV
    df.to_csv(csv_filename, index=False)
    print(f"Created simulated event data: {csv_filename}")
    
    # Load data using data_loader
    bin_minutes = 30  # 30-minute bins
    binary_data, bins_per_day = load_binary_from_csv(
        csv_filename,
        timestamp_col="timestamp",
        value_col="value",
        value_threshold=0.5,  # Only count high-value events
        bin_minutes=bin_minutes,
        start_hour=0
    )
    
    print(f"Loaded {binary_data.sum()} events in {len(binary_data)} bins ({bins_per_day} bins per day)")
    
    # Create BOCPD model with time-of-day hazard
    # Boost hazard at 8am, 12pm, 5pm (typical transition times)
    day_indices = np.array([8*60, 12*60, 17*60]) // bin_minutes
    boundary_indices = frozenset(day_indices)
    
    hazard = BoostedBoundaryHazard(
        base=ConstantHazard(mean_run_length=bins_per_day/2),
        period=bins_per_day,
        boundary_indices=boundary_indices,
        boost_factor=5.0
    )
    
    model = BOCPD(hazard)
    result = model.run(binary_data)
    
    # Create time axis for plotting
    bin_times = [start_date + timedelta(minutes=i*bin_minutes) for i in range(len(binary_data))]
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot binary data
    axes[0].step(bin_times, binary_data, where='post')
    axes[0].set_ylabel('Event')
    axes[0].set_title('Binned Event Data (30-min intervals)')
    axes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M'))
    
    # Plot changepoint probability
    axes[1].plot(bin_times, result.cp_prob)
    axes[1].set_ylabel('CP Probability')
    axes[1].set_title('Changepoint Probability')
    axes[1].set_xlabel('Time')
    axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M'))
    
    # Mark expected daily pattern changes (8am, 12pm, 5pm)
    for day in range(5):
        for hour, minute in [(8, 0), (12, 0), (17, 0)]:
            cp_time = start_date + timedelta(days=day, hours=hour, minutes=minute)
            axes[0].axvline(x=cp_time, color='r', linestyle='--', alpha=0.5)
            axes[1].axvline(x=cp_time, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("example4_csv_data.png")
    plt.close()
    
    print(f"Plot saved to example4_csv_data.png")


if __name__ == "__main__":
    example_1_basic_usage()
    example_2_online_processing()
    example_3_custom_hazard()
    example_4_real_data()
    
    print("\nAll examples completed successfully!")
