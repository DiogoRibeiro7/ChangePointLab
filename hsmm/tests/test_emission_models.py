import numpy as np

def test_emission_models():
    """Test the new emission models (full covariance Gaussian and AR)."""
    print("\n" + "=" * 60)
    print("TESTING EMISSION MODELS")
    print("=" * 60)

    # Generate test data
    np.random.seed(42)
    T, D, K = 500, 4, 3

    # Create synthetic HMM data with different emission types
    print(f"Generating {T} samples of {D}-dimensional data with {K} states")

    # True state sequence (simple alternating pattern)
    true_states = np.random.choice(K, size=T)

    # Test 1: Full Covariance Gaussian Emissions
    print(f"\n--- Testing Full Covariance Gaussian Emissions ---")

    from hsmm.gaussian_full import GaussianFullEmissions

    # Generate data with full covariance structure
    true_means = np.random.normal(0, 2, size=(K, D))
    true_covs = np.zeros((K, D, D))

    for k in range(K):
        # Generate random PSD matrix
        A = np.random.normal(0, 1, size=(D, D))
        true_covs[k] = A @ A.T + 0.1 * np.eye(D)

    # Generate observations
    X_gaussian = np.zeros((T, D))
    for t in range(T):
        k = true_states[t]
        X_gaussian[t] = np.random.multivariate_normal(true_means[k], true_covs[k])

    # Test emission model
    gaussian_emissions = GaussianFullEmissions(n_states=K)

    # Test k-means initialization
    gaussian_emissions.initialize_kmeans(X_gaussian, n_init=3, seed=42)
    loglik_gaussian = gaussian_emissions.compute_loglik(X_gaussian)

    print(f"✓ Full covariance Gaussian emissions initialized")
    print(f"  Log-likelihood shape: {loglik_gaussian.shape}")
    print(f"  Mean log-likelihood: {loglik_gaussian.mean():.4f}")
    print(f"  Number of parameters: {gaussian_emissions.n_parameters}")

    # Test parameter updates
    responsibilities = np.random.dirichlet([1] * K, size=T)  # Random soft assignments
    gaussian_emissions.update_from_responsibilities(X_gaussian, responsibilities)
    new_loglik = gaussian_emissions.compute_loglik(X_gaussian)

    print(f"  After parameter update: {new_loglik.mean():.4f}")

    # Test 2: AR Emissions
    print(f"\n--- Testing AR Emissions ---")

    from hsmm.ar_emissions import AREmissions

    # Generate AR(2) data
    ar_order = 2
    ar_emissions_true = AREmissions(n_states=K, order=ar_order)

    # Initialize with known structure
    ar_emissions_true.initialize(X_gaussian, method="kmeans", seed=42)

    # Generate AR observations
    X_ar = ar_emissions_true.sample(true_states, seed=42)

    print(f"✓ Generated AR({ar_order}) data")

    # Test AR emission estimation
    ar_emissions_test = AREmissions(n_states=K, order=ar_order)

    for method in ["random", "global", "kmeans"]:
        print(f"  Testing {method} initialization:")
        ar_emissions_test.initialize(X_ar, method=method, seed=42)
        loglik_ar = ar_emissions_test.compute_loglik(X_ar)

        print(f"    Log-likelihood: {loglik_ar[ar_order:].mean():.4f}")  # Skip first p points
        print(f"    Parameters: {ar_emissions_test.n_parameters}")

    # Test parameter updates with responsibilities
    ar_emissions_test.update_from_responsibilities(X_ar, responsibilities)
    updated_loglik = ar_emissions_test.compute_loglik(X_ar)
    print(f"  After soft update: {updated_loglik[ar_order:].mean():.4f}")

    print("Emission model testing completed")
