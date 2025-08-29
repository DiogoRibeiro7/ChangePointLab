import numpy as np

def test_bandwidth_cv():
    """Test automatic bandwidth cross-validation."""
    print("\n" + "=" * 60)
    print("TESTING BANDWIDTH CROSS-VALIDATION")
    print("=" * 60)

    # Generate test data with known optimal scale
    np.random.seed(42)
    n, d = 400, 3

    # Create well-separated clusters (optimal bandwidth should be around 1-2)
    cluster1 = np.random.normal([0, 0, 0], 0.5, size=(n // 2, d))
    cluster2 = np.random.normal([4, 4, 4], 0.5, size=(n // 2, d))
    X = np.vstack([cluster1, cluster2])

    print(f"Generated {n} samples with 2 well-separated clusters")

    from kcp.bandwidth_cv import (
        select_rbf_bandwidth_cv,
        select_rbf_bandwidth_information_criterion,
        select_rbf_bandwidth_multiscale,
        bandwidth_stability_analysis,
        _median_heuristic,
    )

    # Baseline: median heuristic
    sigma_median = _median_heuristic(X)
    print(f"Median heuristic baseline: σ = {sigma_median:.4f}")

    print(f"\n--- Cross-Validation Methods ---")

    # Test 1: K-fold CV
    print("K-fold cross-validation:")
    sigma_kfold = select_rbf_bandwidth_cv(
        X, cv_folds=5, method="kfold", scoring="likelihood", n_candidates=15
    )
    print(
        f"✓ K-fold CV selected: σ = {sigma_kfold:.4f} (ratio to median: {sigma_kfold / sigma_median:.2f})"
    )

    # Test 2: Time series CV (treating data as sequential)
    print("Time series cross-validation:")
    sigma_ts = select_rbf_bandwidth_cv(
        X, cv_folds=5, method="timeseries", scoring="likelihood", n_candidates=15
    )
    print(
        f"✓ Time series CV selected: σ = {sigma_ts:.4f} (ratio to median: {sigma_ts / sigma_median:.2f})"
    )

    print(f"\n--- Information Criteria ---")

    # Test 3: BIC selection
    sigma_bic = select_rbf_bandwidth_information_criterion(X, criterion="bic", n_candidates=15)
    print(f"✓ BIC selected: σ = {sigma_bic:.4f} (ratio to median: {sigma_bic / sigma_median:.2f})")

    # Test 4: AIC selection
    sigma_aic = select_rbf_bandwidth_information_criterion(X, criterion="aic", n_candidates=15)
    print(f"✓ AIC selected: σ = {sigma_aic:.4f} (ratio to median: {sigma_aic / sigma_median:.2f})")

    print(f"\n--- Multi-Scale Analysis ---")

    # Test 5: Multi-scale selection
    multiscale_results = select_rbf_bandwidth_multiscale(X, n_scales=3, base_method="cv")
    print("Multi-scale bandwidth selection:")
    for scale, sigma in multiscale_results.items():
        print(f"  {scale}: σ = {sigma:.4f} (ratio to median: {sigma / sigma_median:.2f})")

    print(f"\n--- Stability Analysis ---")

    # Test 6: Bootstrap stability
    stability = bandwidth_stability_analysis(X, n_bootstrap=20, subsample_ratio=0.8)
    print("Bandwidth selection stability:")
    print(f"  Mean: {stability['mean']:.4f} ± {stability['std']:.4f}")
    print(f"  Median: {stability['median']:.4f}")
    print(f"  25th-75th percentile: [{stability['q25']:.4f}, {stability['q75']:.4f}]")
    print(f"  Coefficient of variation: {stability['coefficient_of_variation']:.3f}")

    # Stability assessment
    if stability["coefficient_of_variation"] < 0.2:
        print("  → Selection is STABLE")
    elif stability["coefficient_of_variation"] < 0.5:
        print("  → Selection is MODERATELY STABLE")
    else:
        print("  → Selection is UNSTABLE")

    print("Bandwidth cross-validation testing completed")


