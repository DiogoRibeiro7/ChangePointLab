import numpy as np

def test_rff_variants():
    """Test the advanced RFF variants."""
    print("\n" + "=" * 60)
    print("TESTING RFF VARIANTS")
    print("=" * 60)

    # Generate test data with multiple scales
    np.random.seed(42)
    n, d = 800, 6

    # Multi-scale data
    X1 = np.random.normal([0, 0, 0, 0, 0, 0], 0.5, size=(n // 3, d))
    X2 = np.random.normal([3, 3, 3, 3, 3, 3], 1.0, size=(n // 3, d))
    X3 = np.random.normal([-2, 2, -1, 1, -2, 2], 1.5, size=(n // 3, d))
    X = np.vstack([X1, X2, X3])

    print(f"Generated {n} samples in {d} dimensions")

    from kcp.rff_variants import (
        OrthogonalRFFConfig,
        QuasiMCRFFConfig,
        CompactRFFConfig,
        orthogonal_rff_map,
        quasi_mc_rff_map,
        compact_support_rff_map,
        compare_rff_variants,
        adaptive_rff_map,
    )

    n_features = 256

    print(f"\n--- Testing RFF Variants ---")

    # Test 1: Orthogonal RFF
    print("Testing Orthogonal RFF:")
    orth_config = OrthogonalRFFConfig(n_features=n_features, structured=True, seed=42)
    orth_rff = orthogonal_rff_map(X, orth_config)

    print(f"✓ Orthogonal RFF: {orth_rff.Z.shape}, γ={orth_rff.gamma:.4f}")

    # Test 2: Quasi-Monte Carlo RFF
    print("Testing Quasi-MC RFF variants:")
    for seq_type in ["sobol", "halton", "latin_hypercube"]:
        qmc_config = QuasiMCRFFConfig(n_features=n_features, sequence_type=seq_type, seed=42)
        qmc_rff = quasi_mc_rff_map(X, qmc_config)
        print(f"✓ QMC-RFF ({seq_type}): {qmc_rff.Z.shape}, γ={qmc_rff.gamma:.4f}")

    # Test 3: Compact Support RFF
    print("Testing Compact Support RFF:")
    compact_config = CompactRFFConfig(n_features=n_features, support_radius=2.0, seed=42)
    compact_rff = compact_support_rff_map(X, compact_config)

    print(f"✓ Compact RFF: {compact_rff.Z.shape}, γ={compact_rff.gamma:.4f}")

    # Test 4: Comparative Analysis
    print(f"\n--- RFF Variant Comparison ---")
    comparison = compare_rff_variants(X, n_features=n_features, seed=42)

    print("Kernel approximation quality:")
    for variant, metrics in comparison.items():
        print(
            f"  {variant:15s}: MSE={metrics['mse']:.6f}, "
            f"Frobenius={metrics['frobenius_relative_error']:.4f}"
        )

    # Test 5: Adaptive RFF
    print(f"\n--- Adaptive RFF Selection ---")
    adaptive_rff = adaptive_rff_map(X, base_features=128, max_features=512, tolerance=1e-3, seed=42)

    print(f"✓ Adaptive RFF selected {adaptive_rff.config['final_features']} features")
    print(f"  Final approximation error: {adaptive_rff.config['approx_error']:.6f}")

    print("RFF variants testing completed")

