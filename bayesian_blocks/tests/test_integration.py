#!/usr/bin/env python3
"""
Integration test script to verify the fixed Bayesian Blocks implementation works correctly.
Run this after implementing the fixes to ensure everything integrates properly.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List


def test_basic_functionality():
    """Test that all basic functions work."""
    print("=" * 60)
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 60)

    # Import the fixed modules
    try:
        from bayesian_blocks import (
            bayesian_blocks_events,
            bayesian_blocks_counts,
            bayesian_blocks_bernoulli,
            bayesian_blocks,
            BBConfig,
            DataType,
        )

        print("✅ Successfully imported core functions")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    rng = np.random.default_rng(42)

    # Test 1: Counts
    print("\nTesting bayesian_blocks_counts...")
    try:
        count_data = rng.poisson(2.0, 100)
        result = bayesian_blocks_counts(count_data, p0=0.05)
        assert len(result.block_value) > 0
        assert len(result.edges) == len(result.block_value) + 1
        assert hasattr(result, "aic")  # Enhanced result
        print(f"✅ Counts: {len(result.block_value)} blocks detected")
    except Exception as e:
        print(f"❌ Counts test failed: {e}")
        return False

    # Test 2: Events
    print("Testing bayesian_blocks_events...")
    try:
        event_data = np.cumsum(rng.exponential(0.5, 50))
        result = bayesian_blocks_events(event_data, p0=0.05)
        assert len(result.block_value) > 0
        print(f"✅ Events: {len(result.block_value)} blocks detected")
    except Exception as e:
        print(f"❌ Events test failed: {e}")
        return False

    # Test 3: Bernoulli
    print("Testing bayesian_blocks_bernoulli...")
    try:
        binary_data = rng.binomial(1, 0.3, 100)
        result = bayesian_blocks_bernoulli(binary_data, p0=0.05)
        assert len(result.block_value) > 0
        assert np.all(result.block_value >= 0)
        assert np.all(result.block_value <= 1)
        print(f"✅ Bernoulli: {len(result.block_value)} blocks detected")
    except Exception as e:
        print(f"❌ Bernoulli test failed: {e}")
        return False

    # Test 4: Unified API
    print("Testing unified API...")
    try:
        # Auto-detection
        result1 = bayesian_blocks(count_data, data_type="auto")
        result2 = bayesian_blocks(binary_data, data_type="auto")
        result3 = bayesian_blocks(event_data, data_type="auto")

        # Explicit types
        result4 = bayesian_blocks(count_data, data_type="counts")
        result5 = bayesian_blocks(binary_data, data_type="bernoulli")
        result6 = bayesian_blocks(event_data, data_type="events")

        assert all(
            len(r.block_value) > 0
            for r in [result1, result2, result3, result4, result5, result6]
        )
        print("✅ Unified API working correctly")
    except Exception as e:
        print(f"❌ Unified API test failed: {e}")
        return False

    # Test 5: Configuration objects
    print("Testing BBConfig...")
    try:
        config = BBConfig(p0=0.01, min_block_size=2)
        result = bayesian_blocks(count_data, data_type="counts", config=config)
        assert result.config == config
        print("✅ Configuration objects working")
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

    print("\n🎉 All basic functionality tests passed!")
    return True


def test_plotting_integration():
    """Test plotting functionality."""
    print("\n" + "=" * 60)
    print("TESTING PLOTTING INTEGRATION")
    print("=" * 60)

    try:
        from bayesian_blocks import bayesian_blocks_counts, bayesian_blocks_events
        from bb_plotting import plot_blocks_time, plot_blocks_index, BBPlotter

        print("✅ Successfully imported plotting functions")
    except ImportError as e:
        print(f"❌ Plotting import error: {e}")
        return False

    rng = np.random.default_rng(42)

    # Test index-based plotting
    print("\nTesting index-based plotting...")
    try:
        count_data = rng.poisson(2.0, 100)
        result = bayesian_blocks_counts(count_data, p0=0.05)

        fig, ax = plt.subplots(figsize=(8, 4))
        plot_blocks_index(N=len(count_data), result=result, ax=ax)
        plt.close(fig)
        print("✅ Index plotting successful")
    except Exception as e:
        print(f"❌ Index plotting failed: {e}")
        return False

    # Test time-based plotting
    print("Testing time-based plotting...")
    try:
        event_data = np.cumsum(rng.exponential(0.5, 50))
        result = bayesian_blocks_events(event_data, t_start=0.0, t_stop=event_data[-1])

        fig, ax = plt.subplots(figsize=(8, 4))
        plot_blocks_time(t_min=0.0, t_max=event_data[-1], result=result, ax=ax)
        plt.close(fig)
        print("✅ Time plotting successful")
    except Exception as e:
        print(f"❌ Time plotting failed: {e}")
        return False

    # Test advanced plotting
    print("Testing BBPlotter class...")
    try:
        plotter = BBPlotter(result, event_data)
        fig, ax = plt.subplots()
        plotter.plot_blocks(ax=ax)
        plt.close(fig)
        print("✅ BBPlotter working")
    except Exception as e:
        print(f"❌ BBPlotter failed: {e}")
        return False

    print("\n🎉 All plotting tests passed!")
    return True


def test_advanced_features():
    """Test advanced utilities."""
    print("\n" + "=" * 60)
    print("TESTING ADVANCED FEATURES")
    print("=" * 60)

    try:
        from bb_utils import (
            StreamingBayesianBlocks,
            AdaptiveBayesianBlocks,
            detect_outlier_blocks,
        )
        from bayesian_blocks import BBConfig, BBResult

        print("✅ Successfully imported advanced utilities")
    except ImportError as e:
        print(f"❌ Advanced utilities import error: {e}")
        return False

    rng = np.random.default_rng(42)

    # Test streaming
    print("\nTesting streaming algorithm...")
    try:
        config = BBConfig(p0=0.05)
        streaming = StreamingBayesianBlocks(config, buffer_size=25)

        # Add data incrementally
        for i in range(5):
            batch = rng.poisson(2.0, 10).tolist()
            result = streaming.update(batch)

        final_result = streaming.finalize()
        assert final_result is not None
        print("✅ Streaming algorithm working")
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        return False

    # Test adaptive algorithm
    print("Testing adaptive algorithm...")
    try:
        adaptive = AdaptiveBayesianBlocks()
        data = rng.normal(0, 2, 200)
        result = adaptive.fit(data)
        assert result is not None
        assert len(adaptive.adaptation_history) == 1
        print("✅ Adaptive algorithm working")
    except Exception as e:
        print(f"❌ Adaptive test failed: {e}")
        return False

    # Test outlier detection
    print("Testing outlier detection...")
    try:
        # Create artificial result with outlier
        edges = np.array([0, 10, 20, 30, 40])
        values = np.array([1.0, 1.0, 10.0, 1.0])  # Outlier at index 2
        cps = np.array([10, 20, 30])

        result = BBResult(edges=edges, block_value=values, change_points=cps)
        outlier_indices, scores = detect_outlier_blocks(result, threshold=2.0)

        assert 2 in outlier_indices  # Should detect the outlier
        print("✅ Outlier detection working")
    except Exception as e:
        print(f"❌ Outlier detection failed: {e}")
        return False

    print("\n🎉 All advanced feature tests passed!")
    return True


def test_original_examples():
    """Test that original example scripts would work."""
    print("\n" + "=" * 60)
    print("TESTING ORIGINAL EXAMPLE COMPATIBILITY")
    print("=" * 60)

    rng = np.random.default_rng(0)  # Use same seed as examples

    # Test binned_counts.py equivalent
    print("Testing binned counts example...")
    try:
        from bayesian_blocks import bayesian_blocks_counts
        from bb_plotting import plot_blocks_index

        N = 200
        rate = np.r_[np.full(80, 3.0), np.full(120, 0.8)]
        counts = rng.poisson(rate)
        res = bayesian_blocks_counts(counts, widths=None, p0=0.05)

        fig, ax = plt.subplots(figsize=(8, 4))
        plot_blocks_index(N=N, result=res, ylabel="rate", title="Binned Poisson", ax=ax)
        plt.close(fig)
        print("✅ Binned counts example works")
    except Exception as e:
        print(f"❌ Binned counts example failed: {e}")
        return False

    # Test bernoulli_binary_stream.py equivalent
    print("Testing Bernoulli example...")
    try:
        from bayesian_blocks import bayesian_blocks_bernoulli

        N = 300
        p = np.r_[np.full(120, 0.2), np.full(180, 0.6)]
        x = rng.binomial(1, p)

        res = bayesian_blocks_bernoulli(successes=x, trials=None, p0=0.05)

        fig, ax = plt.subplots(figsize=(8, 4))
        plot_blocks_index(N=N, result=res, ylabel="p", title="Bernoulli blocks", ax=ax)
        plt.close(fig)
        print("✅ Bernoulli example works")
    except Exception as e:
        print(f"❌ Bernoulli example failed: {e}")
        return False

    # Test event_times.py equivalent
    print("Testing event times example...")
    try:
        from bayesian_blocks import bayesian_blocks_events
        from bb_plotting import plot_blocks_time

        # piecewise-constant rate: 2.0 until t=5, then 0.5 until t=10
        t1 = np.cumsum(rng.exponential(1 / 2.0, size=120))
        t1 = t1[t1 <= 5.0]
        t2 = 5.0 + np.cumsum(rng.exponential(1 / 0.5, size=120))
        t2 = t2[t2 <= 10.0]
        t = np.sort(np.concatenate([t1, t2]))

        res = bayesian_blocks_events(t, t_start=0.0, t_stop=10.0, p0=0.05)

        fig, ax = plt.subplots(figsize=(8, 4))
        plot_blocks_time(
            t_min=0.0, t_max=10.0, result=res, title="Events: Poisson rate", ax=ax
        )
        plt.close(fig)
        print("✅ Event times example works")
    except Exception as e:
        print(f"❌ Event times example failed: {e}")
        return False

    print("\n🎉 All original example compatibility tests passed!")
    return True


def test_edge_cases():
    """Test various edge cases."""
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)

    from bayesian_blocks import (
        bayesian_blocks_counts,
        bayesian_blocks_events,
        bayesian_blocks_bernoulli,
    )

    # Test empty data
    print("Testing empty data...")
    try:
        result = bayesian_blocks_counts([])
        assert len(result.edges) == 0
        assert len(result.block_value) == 0

        result = bayesian_blocks_events([])
        assert len(result.edges) == 0

        result = bayesian_blocks_bernoulli([])
        assert len(result.edges) == 0
        print("✅ Empty data handled correctly")
    except Exception as e:
        print(f"❌ Empty data test failed: {e}")
        return False

    # Test single point
    print("Testing single data points...")
    try:
        result = bayesian_blocks_counts([5.0])
        assert len(result.block_value) == 1
        assert result.block_value[0] == 5.0

        result = bayesian_blocks_events([1.0])
        assert len(result.block_value) == 1

        result = bayesian_blocks_bernoulli([1])
        assert len(result.block_value) == 1
        assert result.block_value[0] == 1.0
        print("✅ Single point data handled correctly")
    except Exception as e:
        print(f"❌ Single point test failed: {e}")
        return False

    # Test extreme penalties
    print("Testing extreme penalty values...")
    try:
        rng = np.random.default_rng(42)
        data = rng.poisson(2.0, 100)

        # Very conservative (should give few blocks)
        result1 = bayesian_blocks_counts(data, p0=0.999)

        # Very liberal (should give many blocks)
        result2 = bayesian_blocks_counts(data, p0=1e-6)

        # Should be different
        assert len(result1.block_value) <= len(result2.block_value)
        print("✅ Extreme penalties handled correctly")
    except Exception as e:
        print(f"❌ Extreme penalty test failed: {e}")
        return False

    print("\n🎉 All edge case tests passed!")
    return True


def run_full_test_suite():
    """Run the complete test suite."""
    print("🚀 STARTING BAYESIAN BLOCKS INTEGRATION TEST SUITE")
    print("=" * 70)

    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Plotting Integration", test_plotting_integration),
        ("Advanced Features", test_advanced_features),
        ("Original Examples", test_original_examples),
        ("Edge Cases", test_edge_cases),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} tests...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: CRASHED - {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<25} {status}")

    print(f"\nOverall: {passed}/{total} test suites passed")

    if passed == total:
        print(
            "\n🎉 ALL TESTS PASSED! Your Bayesian Blocks implementation is working correctly."
        )
        print("\nNext steps:")
        print("1. Run the full pytest suite for detailed testing")
        print("2. Try your original example scripts")
        print("3. Explore the new advanced features")
        return True
    else:
        print(
            f"\n⚠️  {total - passed} test suite(s) failed. Please check the errors above."
        )
        print("\nTroubleshooting:")
        print("1. Ensure all files are in the correct locations")
        print("2. Check that imports are working correctly")
        print("3. Verify that the core implementations are complete")
        return False


if __name__ == "__main__":
    success = run_full_test_suite()
    exit(0 if success else 1)
