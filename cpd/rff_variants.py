# rff_variants.py
# MIT License
"""
Advanced Random Fourier Feature variants for improved approximation quality.

Implements:
1. Orthogonal Random Fourier Features (ORFF) - structured random features
2. Quasi-Monte Carlo RFF (QMC-RFF) - low-discrepancy sequences
3. Compact support variants for local kernels

References:
- Yu et al. (2016): Orthogonal Random Features
- Avron et al. (2017): Random Fourier Features for Kernel Ridge Regression
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class OrthogonalRFFConfig:
    """Configuration for Orthogonal Random Fourier Features."""

    n_features: int = 512
    seed: Optional[int] = None
    structured: bool = True  # Use structured orthogonal matrices


@dataclass(frozen=True)
class QuasiMCRFFConfig:
    """Configuration for Quasi-Monte Carlo Random Fourier Features."""

    n_features: int = 512
    seed: Optional[int] = None
    sequence_type: str = "sobol"  # 'sobol', 'halton', 'latin_hypercube'


@dataclass(frozen=True)
class CompactRFFConfig:
    """Configuration for Compact Support Random Fourier Features."""

    n_features: int = 512
    support_radius: float = 1.0  # Kernel becomes zero beyond this radius
    seed: Optional[int] = None


@dataclass(frozen=True)
class AdvancedRFFMap:
    """Advanced RFF mapping with additional metadata."""

    Z: NDArray[np.floating]  # (n, n_features) mapped features
    W: NDArray[np.floating]  # (n_features, d) frequency matrix
    b: NDArray[np.floating]  # (n_features,) phase shifts
    gamma: float  # kernel bandwidth parameter
    variant: str  # variant type identifier
    config: dict  # configuration parameters


def _generate_orthogonal_matrix(
    size: int, structured: bool = True, rng: np.random.Generator = None
) -> NDArray[np.floating]:
    """
    Generate orthogonal matrix for structured random features.

    Parameters
    ----------
    size : int
        Matrix size (will be made square)
    structured : bool
        If True, use structured Hadamard-like construction. Otherwise use QR decomposition.
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    Q : NDArray
        Orthogonal matrix of shape (size, size)
    """
    if rng is None:
        rng = np.random.default_rng()

    if structured and size > 1:
        # Use structured orthogonal construction based on Hadamard-like patterns
        # Find next power of 2
        next_pow2 = 2 ** int(np.ceil(np.log2(size)))

        # Start with Hadamard matrix (or Walsh functions)
        def hadamard_recursive(n):
            if n == 1:
                return np.array([[1]], dtype=float)
            else:
                H_half = hadamard_recursive(n // 2)
                top = np.hstack([H_half, H_half])
                bottom = np.hstack([H_half, -H_half])
                return np.vstack([top, bottom])

        if next_pow2 >= 2:
            H = hadamard_recursive(next_pow2)
            H = H / np.sqrt(next_pow2)  # Normalize

            # Add random diagonal scaling and permutation
            D = rng.choice([-1, 1], size=next_pow2).astype(float)
            P = rng.permutation(np.eye(next_pow2))

            Q_full = P @ np.diag(D) @ H

            # Extract submatrix if needed
            if size < next_pow2:
                Q = Q_full[:size, :size]
            else:
                Q = Q_full
        else:
            Q = np.array([[1]], dtype=float)

    else:
        # QR decomposition of random matrix
        A = rng.normal(0, 1, size=(size, size))
        Q, _ = np.linalg.qr(A)

    return Q


def _sobol_sequence(
    n_points: int, dimension: int, seed: Optional[int] = None
) -> NDArray[np.floating]:
    """
    Generate Sobol low-discrepancy sequence (simplified implementation).

    Note: This is a basic implementation. For production use, consider
    using specialized libraries like scipy.stats.qmc.Sobol.
    """
    if seed is not None:
        np.random.seed(seed)

    # For simplicity, we'll use a pseudo-Sobol based on bit-reversal
    # Real Sobol sequences require direction numbers and more complex construction

    points = np.zeros((n_points, dimension), dtype=float)

    for d in range(dimension):
        # Van der Corput sequence in base 2 for dimension d+2
        base = 2 + d
        for i in range(n_points):
            n_i = i + 1  # 1-indexed
            result = 0.0
            f = 1.0 / base
            while n_i > 0:
                result += f * (n_i % base)
                n_i //= base
                f /= base
            points[i, d] = result

    return points


def _halton_sequence(
    n_points: int, dimension: int, seed: Optional[int] = None
) -> NDArray[np.floating]:
    """Generate Halton low-discrepancy sequence."""

    def van_der_corput(n, base):
        """Van der Corput sequence in given base."""
        result = 0.0
        f = 1.0 / base
        i = n
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result

    # Use first 'dimension' prime numbers as bases
    primes = [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
    ]
    if dimension > len(primes):
        raise ValueError(f"Halton sequence only supports up to {len(primes)} dimensions")

    points = np.zeros((n_points, dimension), dtype=float)

    for d in range(dimension):
        base = primes[d]
        for i in range(n_points):
            points[i, d] = van_der_corput(i + 1, base)

    return points


def _latin_hypercube_sequence(
    n_points: int, dimension: int, seed: Optional[int] = None
) -> NDArray[np.floating]:
    """Generate Latin Hypercube sampling points."""
    if seed is not None:
        np.random.seed(seed)

    points = np.zeros((n_points, dimension), dtype=float)

    for d in range(dimension):
        # Divide [0,1] into n_points intervals and sample one point from each
        intervals = np.arange(n_points) / n_points
        jitter = np.random.uniform(0, 1 / n_points, n_points)
        coords = intervals + jitter

        # Permute to ensure Latin Hypercube property
        points[:, d] = np.random.permutation(coords)

    return points


def orthogonal_rff_map(
    X: NDArray[np.floating],
    config: OrthogonalRFFConfig,
    sigma: Optional[float] = None,
    gamma: Optional[float] = None,
) -> AdvancedRFFMap:
    """
    Generate Orthogonal Random Fourier Features mapping.

    Uses structured orthogonal matrices to reduce variance in the kernel approximation
    compared to standard RFF.

    Parameters
    ----------
    X : NDArray
        Input data of shape (n, d)
    config : OrthogonalRFFConfig
        Configuration parameters
    sigma : Optional[float]
        RBF bandwidth (1/sqrt(2*gamma))
    gamma : Optional[float]
        RBF kernel parameter (gamma = 1/(2*sigma^2))

    Returns
    -------
    AdvancedRFFMap
        RFF mapping with orthogonal structure
    """
    n, d = X.shape
    rng = np.random.default_rng(config.seed)

    # Determine bandwidth
    if gamma is not None:
        bandwidth = 1.0 / np.sqrt(2 * gamma)
    elif sigma is not None:
        bandwidth = sigma
        gamma = 1.0 / (2 * sigma**2)
    else:
        # Median heuristic
        if n > 1:
            pairwise_dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
            median_dist = np.median(pairwise_dists[pairwise_dists > 0])
            bandwidth = median_dist
            gamma = 1.0 / (2 * bandwidth**2)
        else:
            bandwidth = 1.0
            gamma = 0.5

    # Generate structured frequency matrix
    n_features = config.n_features

    if config.structured:
        # Use block structure with orthogonal matrices
        # Number of blocks (each of size d x d)
        n_blocks = max(1, n_features // d)
        actual_features = n_blocks * d

        W_blocks = []
        for block_idx in range(n_blocks):
            # Generate orthogonal matrix for this block
            Q = _generate_orthogonal_matrix(d, structured=True, rng=rng)

            # Scale by bandwidth and add random scaling per row
            scales = rng.exponential(1.0, d) / bandwidth  # Chi-distributed scaling
            W_block = np.diag(scales) @ Q
            W_blocks.append(W_block)

        W = np.vstack(W_blocks)  # (actual_features, d)

        # Trim to exact number of features if needed
        if actual_features > n_features:
            W = W[:n_features, :]
        elif actual_features < n_features:
            # Add extra random features
            n_extra = n_features - actual_features
            W_extra = rng.normal(0, 1 / bandwidth, size=(n_extra, d))
            W = np.vstack([W, W_extra])

    else:
        # Use orthogonalized random matrix
        W_raw = rng.normal(0, 1 / bandwidth, size=(n_features, d))

        if n_features >= d:
            # QR decomposition for orthogonalization
            Q_W, R_W = np.linalg.qr(W_raw.T)
            W = (Q_W @ np.diag(np.sign(np.diag(R_W)))).T

            # If we have more features than dimensions, add random orthogonal complement
            if n_features > d:
                # Generate additional orthogonal vectors
                remaining = n_features - d
                W_extra = rng.normal(0, 1 / bandwidth, size=(remaining, d))
                # Orthogonalize against existing W
                for i in range(remaining):
                    for j in range(d):
                        W_extra[i] -= np.dot(W_extra[i], W[j]) * W[j]
                    W_extra[i] /= np.linalg.norm(W_extra[i]) + 1e-12
                W = np.vstack([W, W_extra])
        else:
            W = W_raw

    # Random phase shifts
    b = rng.uniform(0, 2 * np.pi, size=n_features)

    # Compute features
    linear_part = X @ W.T  # (n, n_features)
    Z = np.sqrt(2.0 / n_features) * np.cos(linear_part + b[None, :])

    return AdvancedRFFMap(
        Z=Z,
        W=W,
        b=b,
        gamma=gamma,
        variant="orthogonal",
        config={"structured": config.structured, "n_features": n_features},
    )


def quasi_mc_rff_map(
    X: NDArray[np.floating],
    config: QuasiMCRFFConfig,
    sigma: Optional[float] = None,
    gamma: Optional[float] = None,
) -> AdvancedRFFMap:
    """
    Generate Quasi-Monte Carlo Random Fourier Features mapping.

    Uses low-discrepancy sequences instead of pseudo-random numbers
    to achieve better convergence rates.

    Parameters
    ----------
    X : NDArray
        Input data of shape (n, d)
    config : QuasiMCRFFConfig
        Configuration parameters
    sigma : Optional[float]
        RBF bandwidth
    gamma : Optional[float]
        RBF kernel parameter

    Returns
    -------
    AdvancedRFFMap
        QMC-RFF mapping
    """
    n, d = X.shape

    # Determine bandwidth
    if gamma is not None:
        bandwidth = 1.0 / np.sqrt(2 * gamma)
    elif sigma is not None:
        bandwidth = sigma
        gamma = 1.0 / (2 * sigma**2)
    else:
        # Median heuristic
        if n > 1:
            pairwise_dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
            median_dist = np.median(pairwise_dists[pairwise_dists > 0])
            bandwidth = median_dist
            gamma = 1.0 / (2 * bandwidth**2)
        else:
            bandwidth = 1.0
            gamma = 0.5

    n_features = config.n_features

    # Generate low-discrepancy sequence for frequency sampling
    if config.sequence_type == "sobol":
        uniform_points = _sobol_sequence(n_features, d, config.seed)
    elif config.sequence_type == "halton":
        uniform_points = _halton_sequence(n_features, d, config.seed)
    elif config.sequence_type == "latin_hypercube":
        uniform_points = _latin_hypercube_sequence(n_features, d, config.seed)
    else:
        raise ValueError(f"Unknown sequence type: {config.sequence_type}")

    # Transform uniform points to Gaussian frequencies using inverse CDF
    # For multivariate Gaussian: first transform to standard normal, then scale
    from scipy.stats import norm  # Would need NumPy-only version

    # NumPy-only Box-Muller approximation for normal inverse CDF
    def normal_ppf(u):
        """Approximation to normal percent point function (inverse CDF)."""
        # Beasley-Springer-Moro approximation
        u = np.clip(u, 1e-15, 1 - 1e-15)

        # Constants
        a = [
            0,
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
        b = [
            0,
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
        c = [
            0,
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
        d = [
            0,
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]

        split = 0.5
        result = np.zeros_like(u)

        # Lower tail
        mask_low = u < split
        if np.any(mask_low):
            t = np.sqrt(-2 * np.log(u[mask_low]))
            num = c[0]
            for i in range(1, len(c)):
                num = num * t + c[i]
            den = 1
            for i in range(1, len(d)):
                den = den * t + d[i]
            result[mask_low] = t - num / den

        # Upper tail
        mask_high = u >= split
        if np.any(mask_high):
            t = np.sqrt(-2 * np.log(1 - u[mask_high]))
            num = c[0]
            for i in range(1, len(c)):
                num = num * t + c[i]
            den = 1
            for i in range(1, len(d)):
                den = den * t + d[i]
            result[mask_high] = -(t - num / den)

        return result

    # Transform to Gaussian
    gaussian_points = normal_ppf(uniform_points)  # (n_features, d)

    # Scale by bandwidth
    W = gaussian_points / bandwidth  # (n_features, d)

    # Generate phase shifts using same sequence (shifted)
    if config.sequence_type == "sobol":
        phase_uniform = _sobol_sequence(n_features, 1, config.seed + 12345 if config.seed else None)
    else:
        # For other sequences, use simple offset
        np.random.seed(config.seed + 12345 if config.seed else None)
        phase_uniform = np.random.uniform(0, 1, size=(n_features, 1))

    b = 2 * np.pi * phase_uniform.ravel()

    # Compute features
    linear_part = X @ W.T  # (n, n_features)
    Z = np.sqrt(2.0 / n_features) * np.cos(linear_part + b[None, :])

    return AdvancedRFFMap(
        Z=Z,
        W=W,
        b=b,
        gamma=gamma,
        variant=f"quasi_mc_{config.sequence_type}",
        config={"sequence_type": config.sequence_type, "n_features": n_features},
    )


def compact_support_rff_map(
    X: NDArray[np.floating],
    config: CompactRFFConfig,
    sigma: Optional[float] = None,
    gamma: Optional[float] = None,
) -> AdvancedRFFMap:
    """
    Generate Compact Support Random Fourier Features.

    Uses a kernel with compact support, which can be beneficial for
    local similarity problems and sparse representations.

    Parameters
    ----------
    X : NDArray
        Input data of shape (n, d)
    config : CompactRFFConfig
        Configuration parameters
    sigma : Optional[float]
        Bandwidth parameter
    gamma : Optional[float]
        Kernel parameter

    Returns
    -------
    AdvancedRFFMap
        Compact support RFF mapping
    """
    n, d = X.shape
    rng = np.random.default_rng(config.seed)

    # For compact support kernels, sigma controls the support radius
    if sigma is not None:
        support_radius = sigma
    elif gamma is not None:
        support_radius = 1.0 / np.sqrt(gamma)
    else:
        support_radius = config.support_radius

    gamma = 1.0 / (support_radius**2)
    n_features = config.n_features

    # For compact support RBF: k(x,y) = max(0, 1 - ||x-y||/sigma)^p
    # We'll use p=2 for smoothness. The Fourier transform involves Bessel functions.

    # Approximate compact RBF with truncated Gaussian frequencies
    # Sample from truncated normal distribution
    cutoff = 3.0  # Truncate at 3 standard deviations

    W = np.zeros((n_features, d), dtype=float)
    b = rng.uniform(0, 2 * np.pi, size=n_features)

    for i in range(n_features):
        # Sample frequency vector with proper scaling
        attempts = 0
        while attempts < 100:  # Prevent infinite loop
            w_candidate = rng.normal(0, 1 / support_radius, size=d)
            # Accept if within truncation radius
            if np.linalg.norm(w_candidate) <= cutoff / support_radius:
                W[i] = w_candidate
                break
            attempts += 1

        if attempts >= 100:
            # Fallback: use untruncated sample
            W[i] = rng.normal(0, 1 / support_radius, size=d)

    # Apply compact kernel weighting
    # Weight frequencies by their distance from origin (closer = higher weight)
    freq_norms = np.linalg.norm(W, axis=1)
    weights = np.maximum(0, 1 - freq_norms * support_radius / cutoff) ** 2

    # Compute features with weighting
    linear_part = X @ W.T  # (n, n_features)
    Z = np.sqrt(2.0 / n_features) * weights[None, :] * np.cos(linear_part + b[None, :])

    return AdvancedRFFMap(
        Z=Z,
        W=W,
        b=b,
        gamma=gamma,
        variant="compact_support",
        config={"support_radius": support_radius, "n_features": n_features},
    )


def adaptive_rff_map(
    X: NDArray[np.floating],
    base_features: int = 256,
    max_features: int = 2048,
    tolerance: float = 1e-3,
    sigma: Optional[float] = None,
    gamma: Optional[float] = None,
    seed: Optional[int] = None,
) -> AdvancedRFFMap:
    """
    Adaptive RFF that automatically determines the number of features needed
    for a given approximation quality.

    Parameters
    ----------
    X : NDArray
        Input data of shape (n, d)
    base_features : int
        Starting number of features
    max_features : int
        Maximum allowed features
    tolerance : float
        Convergence tolerance for kernel matrix approximation
    sigma : Optional[float]
        RBF bandwidth
    gamma : Optional[float]
        RBF parameter
    seed : Optional[int]
        Random seed

    Returns
    -------
    AdvancedRFFMap
        Adaptive RFF mapping with optimal feature count
    """
    n, d = X.shape
    rng = np.random.default_rng(seed)

    # Determine bandwidth
    if gamma is not None:
        bandwidth = 1.0 / np.sqrt(2 * gamma)
    elif sigma is not None:
        bandwidth = sigma
        gamma = 1.0 / (2 * sigma**2)
    else:
        # Median heuristic
        if n > 1000:
            # Subsample for efficiency
            idx = rng.choice(n, size=1000, replace=False)
            X_sub = X[idx]
        else:
            X_sub = X

        pairwise_dists = np.linalg.norm(X_sub[:, None, :] - X_sub[None, :, :], axis=2)
        median_dist = np.median(pairwise_dists[pairwise_dists > 0])
        bandwidth = median_dist
        gamma = 1.0 / (2 * bandwidth**2)

    # Start with base features and iteratively add more
    current_features = base_features
    best_approx_error = np.inf
    patience_counter = 0
    max_patience = 3

    # For efficiency, only test on a subset if dataset is large
    if n > 500:
        test_idx = rng.choice(n, size=min(500, n), replace=False)
        X_test = X[test_idx]
    else:
        X_test = X
        test_idx = np.arange(n)

    # Compute true kernel matrix (subset)
    true_K = np.exp(-gamma * np.sum((X_test[:, None, :] - X_test[None, :, :]) ** 2, axis=2))

    while current_features <= max_features:
        # Generate RFF features
        W = rng.normal(0, 1 / bandwidth, size=(current_features, d))
        b = rng.uniform(0, 2 * np.pi, size=current_features)

        linear_part = X_test @ W.T
        Z_test = np.sqrt(2.0 / current_features) * np.cos(linear_part + b[None, :])

        # Approximate kernel matrix
        approx_K = Z_test @ Z_test.T

        # Compute approximation error
        approx_error = np.mean((true_K - approx_K) ** 2)

        print(f"Features: {current_features}, Approx Error: {approx_error:.6f}")

        if approx_error < tolerance:
            print(f"Converged with {current_features} features")
            break

        if approx_error < best_approx_error:
            best_approx_error = approx_error
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at {current_features} features")
            break

        # Increase feature count
        current_features = min(int(current_features * 1.5), max_features)

    # Generate final mapping with full dataset
    final_features = current_features
    W = rng.normal(0, 1 / bandwidth, size=(final_features, d))
    b = rng.uniform(0, 2 * np.pi, size=final_features)

    linear_part = X @ W.T
    Z = np.sqrt(2.0 / final_features) * np.cos(linear_part + b[None, :])

    return AdvancedRFFMap(
        Z=Z,
        W=W,
        b=b,
        gamma=gamma,
        variant="adaptive",
        config={
            "final_features": final_features,
            "tolerance": tolerance,
            "approx_error": best_approx_error,
        },
    )


def compare_rff_variants(
    X: NDArray[np.floating],
    n_features: int = 512,
    sigma: Optional[float] = None,
    seed: Optional[int] = 42,
) -> dict:
    """
    Compare different RFF variants on the same dataset.

    Returns approximation quality metrics for each variant.
    """
    n, d = X.shape

    # Standard RFF baseline
    from kcp_rff import rbf_rff_map, RFFConfig

    standard_config = RFFConfig(n_features=n_features, seed=seed)
    standard_rff = rbf_rff_map(X, standard_config, sigma=sigma)

    # Orthogonal RFF
    orth_config = OrthogonalRFFConfig(n_features=n_features, seed=seed, structured=True)
    orth_rff = orthogonal_rff_map(X, orth_config, sigma=sigma)

    # Quasi-MC RFF
    qmc_config = QuasiMCRFFConfig(n_features=n_features, seed=seed, sequence_type="sobol")
    qmc_rff = quasi_mc_rff_map(X, qmc_config, sigma=sigma)

    # Compact support RFF
    compact_config = CompactRFFConfig(n_features=n_features, seed=seed)
    compact_rff = compact_support_rff_map(X, compact_config, sigma=sigma)

    results = {}

    # Test on subset for efficiency
    if n > 300:
        np.random.seed(seed)
        test_idx = np.random.choice(n, size=300, replace=False)
        X_test = X[test_idx]
    else:
        X_test = X

    # True kernel matrix
    gamma = standard_rff.gamma
    true_K = np.exp(-gamma * np.sum((X_test[:, None, :] - X_test[None, :, :]) ** 2, axis=2))

    # Test each variant
    for name, rff_map in [
        ("standard", standard_rff),
        ("orthogonal", orth_rff),
        ("quasi_mc_sobol", qmc_rff),
        ("compact_support", compact_rff),
    ]:
        # Get features for test data
        if hasattr(rff_map, "Z"):
            if rff_map.Z.shape[0] == X.shape[0]:
                Z_test = rff_map.Z[test_idx] if n > 300 else rff_map.Z
            else:
                # Recompute features for test data
                linear_part = X_test @ rff_map.W.T
                Z_test = np.sqrt(2.0 / rff_map.W.shape[0]) * np.cos(
                    linear_part + rff_map.b[None, :]
                )

        # Approximated kernel
        approx_K = Z_test @ Z_test.T

        # Metrics
        mse = np.mean((true_K - approx_K) ** 2)
        frobenius_error = np.linalg.norm(true_K - approx_K, "fro") / np.linalg.norm(true_K, "fro")
        max_error = np.max(np.abs(true_K - approx_K))

        # Spectral properties
        eigvals_true = np.linalg.eigvalsh(true_K)
        eigvals_approx = np.linalg.eigvalsh(approx_K)
        spectral_error = np.mean(
            (np.sort(eigvals_true)[::-1][:50] - np.sort(eigvals_approx)[::-1][:50]) ** 2
        )

        results[name] = {
            "mse": mse,
            "frobenius_relative_error": frobenius_error,
            "max_absolute_error": max_error,
            "spectral_error": spectral_error,
            "gamma": gamma,
            "n_features": rff_map.W.shape[0] if hasattr(rff_map, "W") else n_features,
        }

    return results


# Example usage and testing
if __name__ == "__main__":
    print("Testing RFF variants...")

    # Generate synthetic data
    np.random.seed(42)
    n, d = 1000, 5

    # Create data with some structure
    X1 = np.random.normal(0, 1, size=(n // 2, d))
    X2 = np.random.normal(2, 1.5, size=(n // 2, d))
    X = np.vstack([X1, X2])

    print(f"Testing on {n} samples in {d} dimensions")

    # Compare variants
    comparison = compare_rff_variants(X, n_features=512, seed=42)

    print("\nRFF Variant Comparison:")
    print("-" * 60)
    for variant, metrics in comparison.items():
        print(
            f"{variant:15s}: MSE={metrics['mse']:.6f}, "
            f"Frobenius={metrics['frobenius_relative_error']:.4f}, "
            f"Spectral={metrics['spectral_error']:.6f}"
        )

    # Test adaptive RFF
    print(f"\nTesting adaptive RFF...")
    adaptive_rff = adaptive_rff_map(
        X, base_features=128, max_features=1024, tolerance=1e-3, seed=42
    )
    print(f"Adaptive RFF selected {adaptive_rff.config['final_features']} features")
    print(f"Final approximation error: {adaptive_rff.config['approx_error']:.6f}")
