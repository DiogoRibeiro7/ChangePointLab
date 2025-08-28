# test_comprehensive.py
# Comprehensive test suite with edge cases, performance, and statistical validation

from __future__ import annotations

import numpy as np
import pytest
import time
import warnings
from unittest.mock import patch, MagicMock
from typing import List, Tuple, Callable

# Import the modules (assuming they're in the same directory)
from bayesian_blocks import (
    bayesian_blocks_events,
    bayesian_blocks_counts,
    bayesian_blocks_bernoulli,
    BBResult,
    BBConfig,
    ncp_prior_from_p0,
)
from improved_bayesian_blocks import bayesian_blocks, DataType, _detect_data_type
from advanced_utils import (
    bootstrap_confidence_intervals,
    cross_validate_parameters,
    StreamingBayesianBlocks,
    AdaptiveBayesianBlocks,
    detect_outlier_blocks,
    quick_analysis,
)


class TestBasicFunctionality:
    """Test core functionality with various data types."""

    def test_empty_data(self):
        """Test behavior with empty inputs."""
        # Empty arrays should return empty results
        result = bayesian_blocks_counts([])
        assert len(result.edges) == 0
        assert len(result.block_value) == 0
        assert len(result.change_points) == 0

        # Events with empty array
        result = bayesian_blocks_events([])
        assert len(result.edges) == 0

    def test_single_point_data(self):
        """Test with single data point."""
        # Single count
        result = bayesian_blocks_counts([5.0])
        assert len(result.block_value) == 1
        assert result.block_value[0] == 5.0

        # Single event
        result = bayesian_blocks_events([1.0])
        assert len(result.block_value) == 1

        # Single binary
        result = bayesian_blocks_bernoulli([1])
        assert len(result.block_value) == 1
        assert result.block_value[0] == 1.0

    def test_identical_data(self):
        """Test with identical values."""
        n = 100
        constant_value = 3.5

        # Identical counts should yield single block
        result = bayesian_blocks_counts([constant_value] * n, p0=0.05)
        assert len(result.block_value) == 1
        assert abs(result.block_value[0] - constant_value) < 1e-10

        # Identical binary values
        result = bayesian_blocks_bernoulli([1] * n, p0=0.05)
        assert len(result.block_value) == 1
        assert result.block_value[0] == 1.0

    def test_extreme_penalty_values(self):
        """Test with extreme p0/gamma values."""
        rng = np.random.default_rng(42)
        data = rng.poisson(2.0, 200)

        # Very small p0 should give many blocks
        result_small_p0 = bayesian_blocks_counts(data, p0=1e-6)

        # Very large p0 should give few blocks
        result_large_p0 = bayesian_blocks_counts(data, p0=0.99)

        assert len(result_small_p0.block_value) > len(result_large_p0.block_value)

        # Direct gamma testing
        result_small_gamma = bayesian_blocks_counts(data, gamma=0.1)
        result_large_gamma = bayesian_blocks_counts(data, gamma=100.0)

        assert len(result_small_gamma.block_value) > len(result_large_gamma.block_value)

    def test_invalid_inputs(self):
        """Test various invalid inputs."""
        # Negative counts
        with pytest.raises(ValueError, match="non-negative"):
            bayesian_blocks_counts([-1, 2, 3])

        # Invalid p0
        with pytest.raises(ValueError, match="p0 must be in"):
            bayesian_blocks_counts([1, 2, 3], p0=1.5)

        with pytest.raises(ValueError, match="p0 must be in"):
            bayesian_blocks_counts([1, 2, 3], p0=0.0)

        # Successes > trials
        with pytest.raises(ValueError, match="successes <= trials"):
            bayesian_blocks_bernoulli([3], [2])

        # Negative widths
        with pytest.raises(ValueError, match="widths.*> 0"):
            bayesian_blocks_counts([1, 2], [-1, 1])

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very large values
        large_data = [1e10, 1e10 + 1, 1e10 + 2]
        result = bayesian_blocks_counts(large_data)
        assert np.isfinite(result.block_value).all()

        # Very small positive values
        small_data = [1e-10, 2e-10, 3e-10]
        result = bayesian_blocks_counts(small_data)
        assert np.isfinite(result.block_value).all()

        # Mixed scales
        mixed_data = [1e-6, 1e6, 1e-6, 1e6]
        result = bayesian_blocks_counts(mixed_data)
        assert np.isfinite(result.block_value).all()


class TestDataTypeDetection:
    """Test automatic data type detection."""

    def test_detect_binary_data(self):
        """Test detection of binary sequences."""
        binary_data = [0, 1, 1, 0, 1]
        detected = _detect_data_type(binary_data)
        assert detected == DataType.BERNOULLI

    def test_detect_count_data(self):
        """Test detection of count data."""
        count_data = [0, 2, 5, 1, 8]  # Non-negative integers
        detected = _detect_data_type(count_data)
        assert detected == DataType.COUNTS

    def test_detect_event_data(self):
        """Test detection of continuous event times."""
        event_data = [1.2, 2.7, 3.1, 5.9]  # Continuous values
        detected = _detect_data_type(event_data)
        assert detected == DataType.EVENTS

    def test_detect_tuple_data(self):
        """Test detection with tuple inputs."""
        # (successes, trials)
        bernoulli_tuple = ([1, 0, 1, 1], [1, 1, 1, 1])
        detected = _detect_data_type(bernoulli_tuple)
        assert detected == DataType.BERNOULLI

        # (counts, widths)
        counts_tuple = ([3, 5, 2], [1.0, 1.5, 0.8])
        detected = _detect_data_type(counts_tuple)
        assert detected == DataType.COUNTS


class TestStatisticalProperties:
    """Test statistical properties and correctness."""

    def test_known_changepoint_detection(self):
        """Test detection of known changepoints."""
        rng = np.random.default_rng(42)

        # Create data with known changepoint at position 100
        n1, n2 = 100, 100
        rate1, rate2 = 5.0, 1.0

        data1 = rng.poisson(rate1, n1)
        data2 = rng.poisson(rate2, n2)
        data = np.concatenate([data1, data2])

        result = bayesian_blocks_counts(data, p0=0.05)

        # Should detect at least one changepoint
        assert len(result.block_value) >= 2

        # First block should have higher rate than last
        assert result.block_value[0] > result.block_value[-1]

        # Changepoint should be roughly near position 100
        if len(result.change_points) > 0:
            closest_cp = min(result.change_points, key=lambda x: abs(x - 100))
            assert abs(closest_cp - 100) < 20  # Allow some tolerance

    def test_no_false_positives_constant(self):
        """Test that constant data doesn't produce false positives."""
        rng = np.random.default_rng(42)

        # Generate from single Poisson distribution
        n = 300
        true_rate = 3.0
        data = rng.poisson(true_rate, n)

        # With strong penalty, should prefer single block
        result = bayesian_blocks_counts(data, p0=0.001)

        # Should detect only one block most of the time
        assert len(result.block_value) <= 2  # Allow occasional split

        # Rate should be close to true rate
        overall_rate = np.mean(data)
        assert abs(result.block_value[0] - overall_rate) < 0.5

    def test_bernoulli_probability_estimation(self):
        """Test Bernoulli probability estimation accuracy."""
        rng = np.random.default_rng(42)

        # Known probability change
        n1, n2 = 150, 150
        p1, p2 = 0.2, 0.8

        x1 = rng.binomial(1, p1, n1)
        x2 = rng.binomial(1, p2, n2)
        x = np.concatenate([x1, x2])

        result = bayesian_blocks_bernoulli(x, p0=0.05)

        # Should detect change
        assert len(result.block_value) >= 2

        # First block probability should be close to p1
        assert abs(result.block_value[0] - p1) < 0.1

        # Last block probability should be close to p2
        assert abs(result.block_value[-1] - p2) < 0.1

    def test_event_rate_estimation(self):
        """Test event rate estimation for Poisson processes."""
        rng = np.random.default_rng(42)

        # Generate two-rate Poisson process
        rate1, rate2 = 3.0, 0.5
        t_change = 5.0
        t_end = 10.0

        # Generate event times
        t1 = np.cumsum(rng.exponential(1 / rate1, 200))
        t1 = t1[t1 < t_change]

        t2 = t_change + np.cumsum(rng.exponential(1 / rate2, 200))
        t2 = t2[t2 < t_end]

        events = np.sort(np.concatenate([t1, t2]))

        result = bayesian_blocks_events(events, t_start=0.0, t_stop=t_end, p0=0.05)

        # Should detect rate change
        assert len(result.block_value) >= 2

        # First rate should be higher
        assert result.block_value[0] > result.block_value[-1]


class TestPerformance:
    """Test performance and scalability."""

    def test_linear_scaling_counts(self):
        """Test that algorithm scales reasonably with data size."""
        rng = np.random.default_rng(42)

        sizes = [100, 500, 1000]
        times = []

        for n in sizes:
            data = rng.poisson(2.0, n)

            start = time.time()
            result = bayesian_blocks_counts(data, p0=0.05)
            elapsed = time.time() - start
            times.append(elapsed)

            # Sanity check that it produced a result
            assert len(result.block_value) > 0

        # Should not be exponential growth
        # (This is a rough check - could be more sophisticated)
        assert times[-1] < 10 * times[0]  # No worse than 10x for 10x data

    @pytest.mark.parametrize("data_size", [10, 100, 1000])
    def test_memory_efficiency(self, data_size):
        """Test that memory usage is reasonable."""
        rng = np.random.default_rng(42)
        data = rng.poisson(3.0, data_size)

        # This is a basic test - more sophisticated memory profiling
        # would require additional tools
        result = bayesian_blocks_counts(data)

        # Result arrays should not be dramatically larger than input
        total_result_size = (
            result.edges.nbytes
            + result.block_value.nbytes
            + result.change_points.nbytes
        )
        input_size = np.asarray(data).nbytes

        # Should not use more than 10x input size
        assert total_result_size < 10 * input_size


class TestAdvancedFeatures:
    """Test advanced utilities and extensions."""

    def test_streaming_updates(self):
        """Test streaming Bayesian Blocks."""
        rng = np.random.default_rng(42)

        config = BBConfig(p0=0.05)
        streaming = StreamingBayesianBlocks(config, buffer_size=50)

        # Add data incrementally
        for i in range(10):
            batch = rng.poisson(2.0, 20).tolist()
            result = streaming.update(batch)

            # Should get result when buffer is full
            if i > 0 and (i * 20) % 50 == 0:
                assert result is not None

        # Final result
        final_result = streaming.finalize()
        assert final_result is not None
        assert len(final_result.block_value) > 0

    def test_cross_validation(self):
        """Test parameter cross-validation."""
        rng = np.random.default_rng(42)

        # Generate data with known structure
        data = np.concatenate([rng.poisson(3.0, 100), rng.poisson(1.0, 100)])

        param_grid = {"p0": [0.01, 0.05, 0.1], "min_block_size": [1, 2]}

        cv_result = cross_validate_parameters(data, param_grid, cv_folds=3)

        assert cv_result.best_config is not None
        assert cv_result.best_score is not None
        assert len(cv_result.all_scores) == 6  # 3 * 2 parameter combinations
        assert len(cv_result.all_configs) == 6

    def test_bootstrap_confidence(self):
        """Test bootstrap confidence intervals."""
        rng = np.random.default_rng(42)

        # Small dataset for speed
        data = rng.poisson(2.0, 50)

        def simple_algorithm(d):
            return bayesian_blocks_counts(d, p0=0.05)

        conf_result = bootstrap_confidence_intervals(
            data, simple_algorithm, n_bootstrap=20, n_jobs=1
        )

        assert conf_result.result is not None
        assert len(conf_result.bootstrap_results) == 20
        assert 0 < conf_result.confidence_level < 1

    def test_outlier_detection(self):
        """Test outlier block detection."""
        # Create result with obvious outlier
        edges = np.array([0, 10, 20, 30, 40])
        values = np.array([1.0, 1.0, 10.0, 1.0])  # Middle block is outlier
        cps = np.array([10, 20, 30])

        result = BBResult(edges=edges, block_value=values, change_points=cps)

        outlier_indices, scores = detect_outlier_blocks(result, threshold=2.0)

        assert 2 in outlier_indices  # Middle block should be detected
        assert scores[2] > scores[0]  # Should have higher score

    def test_adaptive_algorithm(self):
        """Test adaptive parameter selection."""
        rng = np.random.default_rng(42)

        # High variance data
        high_var_data = rng.normal(0, 5, 200)

        adaptive = AdaptiveBayesianBlocks()
        result = adaptive.fit(high_var_data)

        assert result is not None
        assert len(adaptive.adaptation_history) == 1

        # Should have adapted parameters
        history = adaptive.adaptation_history[0]
        assert "data_stats" in history
        assert "config" in history
        assert history["data_stats"]["variance"] > 0


class TestIntegration:
    """Integration tests with real-world scenarios."""

    def test_astronomy_like_data(self):
        """Test with astronomy-like event data."""
        rng = np.random.default_rng(42)

        # Simulate a gamma-ray burst: background + burst + background
        t_burst_start, t_burst_end = 100.0, 120.0
        t_total = 200.0

        # Background rate
        bg_rate = 0.1
        burst_rate = 5.0

        # Generate events
        t_bg1 = np.cumsum(rng.exponential(1 / bg_rate, 500))
        t_bg1 = t_bg1[t_bg1 < t_burst_start]

        t_burst = t_burst_start + np.cumsum(rng.exponential(1 / burst_rate, 200))
        t_burst = t_burst[t_burst < t_burst_end]

        t_bg2 = t_burst_end + np.cumsum(rng.exponential(1 / bg_rate, 500))
        t_bg2 = t_bg2[t_bg2 < t_total]

        events = np.sort(np.concatenate([t_bg1, t_burst, t_bg2]))

        result = bayesian_blocks_events(events, t_start=0.0, t_stop=t_total, p0=0.01)

        # Should detect the burst
        assert len(result.block_value) >= 3

        # Find the highest rate block (should be the burst)
        max_rate_idx = np.argmax(result.block_value)
        burst_block_start = result.edges[max_rate_idx]
        burst_block_end = result.edges[max_rate_idx + 1]

        # Burst block should overlap with true burst time
        assert burst_block_start < t_burst_end and burst_block_end > t_burst_start

    def test_financial_like_data(self):
        """Test with financial time series-like data."""
        rng = np.random.default_rng(42)

        # Simulate regime changes in volatility
        n_total = 500

        # Low volatility period
        low_vol = rng.normal(0, 0.5, 200)

        # High volatility period
        high_vol = rng.normal(0, 2.0, 150)

        # Return to low volatility
        low_vol2 = rng.normal(0, 0.5, 150)

        returns = np.concatenate([low_vol, high_vol, low_vol2])

        # Use absolute returns as "intensity" measure
        intensity = np.abs(returns)

        result = bayesian_blocks_counts(intensity, p0=0.05)

        # Should detect regime changes
        assert len(result.block_value) >= 2

        # Middle period should have higher intensity
        if len(result.block_value) >= 3:
            # Rough check that middle block has higher value
            mid_idx = len(result.block_value) // 2
            assert result.block_value[mid_idx] > result.block_value[0]

    def test_unified_api(self):
        """Test the unified API with different data types."""
        rng = np.random.default_rng(42)

        # Test auto-detection and unified interface
        config = BBConfig(p0=0.05)

        # Binary data
        binary_data = rng.binomial(1, 0.3, 100)
        result1 = bayesian_blocks(binary_data, data_type="auto", config=config)
        assert result1 is not None

        # Count data
        count_data = rng.poisson(2.0, 100)
        result2 = bayesian_blocks(count_data, data_type="auto", config=config)
        assert result2 is not None

        # Event data
        event_data = np.cumsum(rng.exponential(0.5, 50))
        result3 = bayesian_blocks(event_data, data_type="auto", config=config)
        assert result3 is not None

        # All should produce valid results
        for result in [result1, result2, result3]:
            assert len(result.block_value) > 0
            assert len(result.edges) == len(result.block_value) + 1


class TestRobustness:
    """Test robustness to various edge cases and corrupted data."""

    def test_nan_handling(self):
        """Test handling of NaN values."""
        data_with_nan = [1.0, 2.0, np.nan, 3.0, 4.0]

        with pytest.raises(ValueError, match="non-finite"):
            bayesian_blocks_counts(data_with_nan)

    def test_inf_handling(self):
        """Test handling of infinite values."""
        data_with_inf = [1.0, 2.0, np.inf, 3.0, 4.0]

        with pytest.raises(ValueError, match="non-finite"):
            bayesian_blocks_counts(data_with_inf)

    def test_very_long_sequences(self):
        """Test with very long sequences."""
        # This tests memory efficiency and prevents regression
        rng = np.random.default_rng(42)

        # Large but manageable size
        n = 5000
        data = rng.poisson(1.0, n)

        # Should complete without memory issues
        result = bayesian_blocks_counts(data, p0=0.1)  # Use larger p0 for speed
        assert result is not None
        assert len(result.block_value) > 0

    def test_edge_case_widths(self):
        """Test edge cases with custom widths."""
        counts = [1, 2, 3, 4]

        # Very small widths
        small_widths = [1e-10, 1e-10, 1e-10, 1e-10]
        result = bayesian_blocks_counts(counts, small_widths)
        assert np.isfinite(result.block_value).all()

        # Very large widths
        large_widths = [1e10, 1e10, 1e10, 1e10]
        result = bayesian_blocks_counts(counts, large_widths)
        assert np.isfinite(result.block_value).all()


# Property-based testing (requires hypothesis)
try:
    from hypothesis import given, strategies as st, settings
    from hypothesis.extra.numpy import arrays

    class TestPropertyBased:
        """Property-based tests using Hypothesis."""

        @given(
            counts=arrays(
                np.float64,
                shape=st.integers(1, 100),
                elements=st.floats(0, 100, allow_nan=False, allow_infinity=False),
            ),
            p0=st.floats(0.01, 0.99),
        )
        @settings(max_examples=20, deadline=None)
        def test_counts_properties(self, counts, p0):
            """Test properties that should always hold for count data."""
            # Skip if all zeros (edge case)
            if np.sum(counts) == 0:
                return

            result = bayesian_blocks_counts(counts, p0=p0)

            # Basic properties
            assert len(result.edges) == len(result.block_value) + 1
            assert len(result.change_points) <= len(result.block_value)
            assert np.all(result.block_value >= 0)  # Rates should be non-negative
            assert np.all(np.isfinite(result.block_value))

            # Edges should be monotonic
            assert np.all(np.diff(result.edges) >= 0)

            # First and last edges should span data
            assert result.edges[0] == 0
            assert result.edges[-1] == len(counts)

        @given(
            binary=arrays(
                np.int32, shape=st.integers(1, 100), elements=st.integers(0, 1)
            ),
            p0=st.floats(0.01, 0.99),
        )
        @settings(max_examples=20, deadline=None)
        def test_bernoulli_properties(self, binary, p0):
            """Test properties for Bernoulli data."""
            result = bayesian_blocks_bernoulli(binary, p0=p0)

            # Probabilities should be in [0, 1]
            assert np.all(result.block_value >= 0)
            assert np.all(result.block_value <= 1)
            assert np.all(np.isfinite(result.block_value))

            # Basic structure properties
            assert len(result.edges) == len(result.block_value) + 1
            assert result.edges[0] == 0
            assert result.edges[-1] == len(binary)

except ImportError:
    print("Hypothesis not available, skipping property-based tests")


# Benchmark/stress tests
@pytest.mark.benchmark
class TestBenchmarks:
    """Benchmark tests for performance regression detection."""

    def test_benchmark_counts_medium(self, benchmark):
        """Benchmark medium-sized count data."""
        rng = np.random.default_rng(42)
        data = rng.poisson(2.0, 1000)

        result = benchmark(bayesian_blocks_counts, data, p0=0.05)
        assert len(result.block_value) > 0

    def test_benchmark_events_medium(self, benchmark):
        """Benchmark medium-sized event data."""
        rng = np.random.default_rng(42)
        events = np.cumsum(rng.exponential(0.1, 500))

        result = benchmark(
            bayesian_blocks_events, events, t_start=0.0, t_stop=events[-1], p0=0.05
        )
        assert len(result.block_value) > 0


if __name__ == "__main__":
    # Run basic smoke tests if executed directly
    print("Running smoke tests...")

    # Basic functionality
    rng = np.random.default_rng(42)

    # Test counts
    count_data = rng.poisson(2.0, 100)
    result = bayesian_blocks_counts(count_data)
    print(f"Counts: {len(result.block_value)} blocks detected")

    # Test events
    event_data = np.cumsum(rng.exponential(0.5, 50))
    result = bayesian_blocks_events(event_data)
    print(f"Events: {len(result.block_value)} blocks detected")

    # Test Bernoulli
    binary_data = rng.binomial(1, 0.3, 100)
    result = bayesian_blocks_bernoulli(binary_data)
    print(f"Bernoulli: {len(result.block_value)} blocks detected")

    print("All smoke tests passed!")

    # Run with pytest for full suite:
    # pytest test_comprehensive.py -v
    # pytest test_comprehensive.py -m benchmark  # Run only benchmarks
