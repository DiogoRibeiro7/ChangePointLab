# bandwidth_cv.py
# MIT License
"""
Automatic bandwidth selection using cross-validation for kernel methods.

Implements multiple strategies:
1. K-fold cross-validation with change-point detection objectives
2. Time series cross-validation for temporal data
3. Information criteria (AIC/BIC) based selection
4. Leave-one-out efficient approximations
5. Multi-scale bandwidth selection
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BandwidthCVConfig:
    """Configuration for bandwidth cross-validation."""

    method: str = "kfold"  # 'kfold', 'timeseries', 'loo', 'information_criterion'
    cv_folds: int = 5
    search_strategy: str = "grid"  # 'grid', 'random', 'bayesian', 'golden_section'
    n_candidates: int = 20
    sigma_range: Tuple[float, float] = (0.1, 10.0)
    log_space: bool = True
    scoring: str = "likelihood"  # 'likelihood', 'mse', 'custom'
    seed: Optional[int] = None


@dataclass
class CVResult:
    """Cross-validation results."""

    best_sigma: float
    best_score: float
    all_sigmas: NDArray[np.floating]
    all_scores: NDArray[np.floating]
    score_std: NDArray[np.floating]
    method: str


def _median_heuristic(X: NDArray[np.floating], subsample: int = 1000) -> float:
    """Compute median heuristic bandwidth.

    Uses a vectorized pairwise distance computation and optional subsampling to
    avoid the \(O(n^2)\) memory blow-up on large datasets.
    """
    def _pairwise_distances(Y: NDArray[np.floating]) -> NDArray[np.floating]:
        sq = np.sum(Y**2, axis=1, keepdims=True)
        sq_dists = sq + sq.T - 2.0 * (Y @ Y.T)
        np.fill_diagonal(sq_dists, 0.0)
        return np.sqrt(np.maximum(sq_dists, 0.0))

    n = X.shape[0]
    if n > subsample:
        idx = np.random.choice(n, size=subsample, replace=False)
        X = X[idx]

    pairwise_dists = _pairwise_distances(X)
    nonzero_dists = pairwise_dists[pairwise_dists > 0]
    return np.median(nonzero_dists) if nonzero_dists.size else 1.0


def _generate_candidate_sigmas(
    config: BandwidthCVConfig, X: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Generate candidate bandwidth values for search."""
    if config.search_strategy == "grid":
        if config.log_space:
            log_min, log_max = np.log(config.sigma_range[0]), np.log(config.sigma_range[1])
            log_sigmas = np.linspace(log_min, log_max, config.n_candidates)
            return np.exp(log_sigmas)
        else:
            return np.linspace(config.sigma_range[0], config.sigma_range[1], config.n_candidates)

    elif config.search_strategy == "random":
        np.random.seed(config.seed)
        if config.log_space:
            log_min, log_max = np.log(config.sigma_range[0]), np.log(config.sigma_range[1])
            log_sigmas = np.random.uniform(log_min, log_max, config.n_candidates)
            return np.exp(log_sigmas)
        else:
            return np.random.uniform(
                config.sigma_range[0], config.sigma_range[1], config.n_candidates
            )

    elif config.search_strategy == "adaptive_grid":
        # Start with coarse grid, then refine around best candidates
        median_sigma = _median_heuristic(X)

        # Coarse grid around median heuristic
        coarse_range = (median_sigma * 0.1, median_sigma * 10.0)
        n_coarse = config.n_candidates // 2

        if config.log_space:
            log_min, log_max = np.log(coarse_range[0]), np.log(coarse_range[1])
            coarse_sigmas = np.exp(np.linspace(log_min, log_max, n_coarse))
        else:
            coarse_sigmas = np.linspace(coarse_range[0], coarse_range[1], n_coarse)

        # Add fine grid around median
        fine_range = (median_sigma * 0.5, median_sigma * 2.0)
        n_fine = config.n_candidates - n_coarse

        if config.log_space:
            log_min, log_max = np.log(fine_range[0]), np.log(fine_range[1])
            fine_sigmas = np.exp(np.linspace(log_min, log_max, n_fine))
        else:
            fine_sigmas = np.linspace(fine_range[0], fine_range[1], n_fine)

        return np.concatenate([coarse_sigmas, fine_sigmas])

    else:
        raise ValueError(f"Unknown search strategy: {config.search_strategy}")


def _kfold_split(
    n: int, k: int, seed: Optional[int] = None
) -> List[Tuple[NDArray[np.integer], NDArray[np.integer]]]:
    """Generate k-fold cross-validation splits."""
    if seed is not None:
        np.random.seed(seed)

    indices = np.random.permutation(n)
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1

    splits = []
    start = 0
    for fold_size in fold_sizes:
        test_idx = indices[start : start + fold_size]
    splits = []
    start = 0
    for fold_size in fold_sizes:
        test_idx = indices[start : start + fold_size]
        train_idx = np.concatenate([indices[:start], indices[start + fold_size :]])
        splits.append((train_idx, test_idx))
        start += fold_size

    return splits


def _time_series_split(
    n: int, n_splits: int, test_size: float = 0.2
) -> List[Tuple[NDArray[np.integer], NDArray[np.integer]]]:
    """Generate time series cross-validation splits (expanding window)."""
    min_train = int(n * 0.3)  # Minimum training size
    test_size_abs = max(1, int(n * test_size))

    splits = []
    for i in range(n_splits):
        # Expanding window: use progressively more training data
        train_end = min_train + i * (n - min_train - test_size_abs) // (n_splits - 1)
        test_start = train_end
        test_end = min(test_start + test_size_abs, n)

        if test_end <= test_start:
            break

        train_idx = np.arange(train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    return splits


def _compute_kernel_loglikelihood(
    X_train: NDArray[np.floating], X_test: NDArray[np.floating], sigma: float, method: str = "kcp"
) -> float:
    """
    Compute kernel-based log-likelihood for bandwidth evaluation.

    This is a proxy score for change-point detection quality.
    """
    # Build kernel matrices
    gamma = 1.0 / (2 * sigma**2)

    # Train kernel matrix
    K_train = np.exp(-gamma * np.sum((X_train[:, None, :] - X_train[None, :, :]) ** 2, axis=2))

    # Cross kernel matrix (test vs train)
    K_cross = np.exp(-gamma * np.sum((X_test[:, None, :] - X_train[None, :, :]) ** 2, axis=2))

    # Test kernel matrix
    K_test = np.exp(-gamma * np.sum((X_test[:, None, :] - X_test[None, :, :]) ** 2, axis=2))

    # Regularization for numerical stability
    reg = 1e-6
    K_train_reg = K_train + reg * np.eye(K_train.shape[0])

    try:
        # Compute conditional likelihood: p(X_test | X_train)
        # Using kernel density estimation framework

        # Cholesky decomposition for efficiency
        L = np.linalg.cholesky(K_train_reg)

        # Solve for kernel weights
        ones_train = np.ones(X_train.shape[0])
        alpha = np.linalg.solve(L, ones_train)
        alpha = np.linalg.solve(L.T, alpha)

        # Normalize
        normalizer = np.sum(alpha)
        if normalizer > 1e-12:
            alpha = alpha / normalizer

        # Compute test likelihoods
        test_densities = K_cross @ alpha

        # Log-likelihood (avoid log(0))
        test_densities = np.maximum(test_densities, 1e-12)
        loglik = np.sum(np.log(test_densities))

        # Penalty for overfitting (larger sigma = simpler model)
        penalty = 0.5 * np.log(np.trace(K_train_reg))

        return loglik - penalty

    except np.linalg.LinAlgError:
        # Numerical issues - return very poor score
        return -np.inf


def _compute_changepoint_score(
    X_train: NDArray[np.floating], X_test: NDArray[np.floating], sigma: float
) -> float:
    """
    Score based on change-point detection performance.

    Fits a simple kernel CPD on training data and evaluates
    how well the model generalizes to test data.
    """
    # Quick kernel CPD using simplified approach
    from kcp import gram_rbf, build_kernel_prefix, kcp_penalized

    try:
        # Fit on training data
        K_train, gamma_train = gram_rbf(X_train, sigma=sigma)
        prefix_train = build_kernel_prefix(K_train)

        # Use a moderate penalty
        penalty = np.log(X_train.shape[0])
        result_train = kcp_penalized(prefix_train, gamma=penalty, min_size=5, method="op")

        # Evaluate on test data
        K_test, _ = gram_rbf(X_test, sigma=sigma)
        prefix_test = build_kernel_prefix(K_test)

        # Score is negative cost (lower cost = better fit)
        result_test = kcp_penalized(prefix_test, gamma=penalty, min_size=5, method="op")

        # Normalize by data size
        score = -result_test.cost / X_test.shape[0]

        return score if np.isfinite(score) else -np.inf

    except Exception:
        return -np.inf


def select_rbf_bandwidth_cv(
    X: NDArray[np.floating],
    config: Optional[BandwidthCVConfig] = None,
    scoring_func: Optional[Callable] = None,
    **config_kwargs,
) -> float:
    """
    Select RBF bandwidth using cross-validation.

    Parameters
    ----------
    X : NDArray
        Input data of shape (n, d)
    config : Optional[BandwidthCVConfig]
        CV configuration
    scoring_func : Optional[Callable]
        Custom scoring function(X_train, X_test, sigma) -> score

    Returns
    -------
    best_sigma : float
        Optimal bandwidth parameter
    """
    if config is None:
        config = BandwidthCVConfig(**config_kwargs)
    elif config_kwargs:
        config = replace(config, **config_kwargs)

    n, d = X.shape

    # Generate candidate sigmas
    candidate_sigmas = _generate_candidate_sigmas(config, X)

    # Generate CV splits
    if config.method == "kfold":
        cv_splits = _kfold_split(n, config.cv_folds, config.seed)
    elif config.method == "timeseries":
        cv_splits = _time_series_split(n, config.cv_folds)
    elif config.method == "loo":
        # Leave-one-out (expensive for large n)
        if n > 200:
            print(f"Warning: LOO-CV with n={n} may be slow. Consider using k-fold.")
        cv_splits = [
            (np.concatenate([np.arange(i), np.arange(i + 1, n)]), np.array([i])) for i in range(n)
        ]
    else:
        raise ValueError(f"Unknown CV method: {config.method}")

    # Choose scoring function
    if scoring_func is not None:
        score_func = scoring_func
    elif config.scoring == "likelihood":
        score_func = _compute_kernel_loglikelihood
    elif config.scoring == "changepoint":
        score_func = _compute_changepoint_score
    else:
        score_func = _compute_kernel_loglikelihood

    # Evaluate each candidate
    scores = np.zeros((len(candidate_sigmas), len(cv_splits)))

    for i, sigma in enumerate(candidate_sigmas):
        for j, (train_idx, test_idx) in enumerate(cv_splits):
            X_train, X_test = X[train_idx], X[test_idx]
            scores[i, j] = score_func(X_train, X_test, sigma)

    # Aggregate scores across folds
    mean_scores = np.mean(scores, axis=1)
    std_scores = np.std(scores, axis=1)

    # Select best bandwidth
    valid_scores = np.isfinite(mean_scores)
    if not np.any(valid_scores):
        # Fallback to median heuristic
        print("Warning: All CV scores invalid, falling back to median heuristic")
        return _median_heuristic(X)

    valid_indices = np.where(valid_scores)[0]
    best_idx = valid_indices[np.argmax(mean_scores[valid_indices])]

    return float(candidate_sigmas[best_idx])


def select_rbf_bandwidth_information_criterion(
    X: NDArray[np.floating],
    criterion: str = "bic",
    sigma_range: Tuple[float, float] = (0.1, 10.0),
    n_candidates: int = 20,
    method: str = "kernel_ridge",
) -> float:
    """
    Select bandwidth using information criteria (AIC/BIC).

    Parameters
    ----------
    X : NDArray
        Input data
    criterion : str
        'aic' or 'bic'
    sigma_range : Tuple[float, float]
        Range of bandwidths to search
    n_candidates : int
        Number of candidates to evaluate
    method : str
        Method for computing effective degrees of freedom

    Returns
    -------
    best_sigma : float
        Optimal bandwidth
    """
    n, d = X.shape

    # Generate candidates
    log_min, log_max = np.log(sigma_range[0]), np.log(sigma_range[1])
    log_sigmas = np.linspace(log_min, log_max, n_candidates)
    candidate_sigmas = np.exp(log_sigmas)

    scores = []

    for sigma in candidate_sigmas:
        gamma = 1.0 / (2 * sigma**2)

        # Build kernel matrix
        K = np.exp(-gamma * np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))

        # Add regularization
        reg = 1e-6 * np.trace(K) / n
        K_reg = K + reg * np.eye(n)

        try:
            # Compute log-likelihood using kernel density estimation
            # p(x_i | X_{-i}) approximation

            # Leave-one-out prediction using matrix inversion lemma
            K_inv = np.linalg.inv(K_reg)
            diag_K_inv = np.diag(K_inv)

            # Predictive variances (diagonal of conditional covariance)
            pred_vars = 1.0 / np.maximum(diag_K_inv, 1e-12)

            # Log-likelihood approximation
            loglik = -0.5 * np.sum(np.log(2 * np.pi * pred_vars))

            # Effective degrees of freedom (trace of smoother matrix)
            if method == "kernel_ridge":
                # K(K + λI)^{-1} smoother matrix trace
                smoother_trace = np.trace(K @ K_inv)
            elif method == "eigenvalue":
                # Based on eigenvalue decay
                eigvals = np.linalg.eigvalsh(K)
                smoother_trace = np.sum(eigvals / (eigvals + reg))
            else:
                smoother_trace = n  # Conservative estimate

            # Information criterion
            if criterion.lower() == "aic":
                score = loglik - smoother_trace
            elif criterion.lower() == "bic":
                score = loglik - 0.5 * smoother_trace * np.log(n)
            else:
                raise ValueError(f"Unknown criterion: {criterion}")

            scores.append(score)

        except np.linalg.LinAlgError:
            scores.append(-np.inf)

    scores = np.array(scores)
    best_idx = np.argmax(scores)

    return float(candidate_sigmas[best_idx])


def select_rbf_bandwidth_multiscale(
    X: NDArray[np.floating], n_scales: int = 5, base_method: str = "cv", **kwargs
) -> Dict[str, float]:
    """
    Multi-scale bandwidth selection for different resolutions.

    Useful when data has structure at multiple scales.

    Parameters
    ----------
    X : NDArray
        Input data
    n_scales : int
        Number of scales to consider
    base_method : str
        Base selection method ('cv' or 'ic')
    **kwargs : dict
        Additional arguments for base method

    Returns
    -------
    dict
        Mapping from scale names to optimal bandwidths
    """
    n, d = X.shape

    # Define scales based on data quantiles
    pairwise_dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    nonzero_dists = pairwise_dists[pairwise_dists > 0]

    # Multi-scale ranges
    percentiles = np.linspace(10, 90, n_scales)
    scale_values = np.percentile(nonzero_dists, percentiles)

    results = {}

    for i, (percentile, scale_val) in enumerate(zip(percentiles, scale_values)):
        scale_name = f"scale_{int(percentile)}pct"

        # Adjust search range around this scale
        scale_range = (scale_val * 0.2, scale_val * 5.0)

        if base_method == "cv":
            config = BandwidthCVConfig(sigma_range=scale_range, n_candidates=15, **kwargs)
            optimal_sigma = select_rbf_bandwidth_cv(X, config)
        elif base_method == "ic":
            optimal_sigma = select_rbf_bandwidth_information_criterion(
                X, sigma_range=scale_range, n_candidates=15, **kwargs
            )
        else:
            raise ValueError(f"Unknown base method: {base_method}")

        results[scale_name] = optimal_sigma

    return results


def bandwidth_stability_analysis(
    X: NDArray[np.floating],
    n_bootstrap: int = 50,
    subsample_ratio: float = 0.8,
    config: Optional[BandwidthCVConfig] = None,
) -> Dict[str, Union[float, NDArray]]:
    """
    Analyze stability of bandwidth selection via bootstrap.

    Parameters
    ----------
    X : NDArray
        Input data
    n_bootstrap : int
        Number of bootstrap iterations
    subsample_ratio : float
        Fraction of data to sample in each iteration
    config : Optional[BandwidthCVConfig]
        CV configuration

    Returns
    -------
    dict
        Statistics about bandwidth stability
    """
    n, d = X.shape
    subsample_size = int(n * subsample_ratio)

    selected_sigmas = []

    for i in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(n, size=subsample_size, replace=True)
        X_boot = X[idx]

        # Select bandwidth
        try:
            sigma_boot = select_rbf_bandwidth_cv(X_boot, config)
            selected_sigmas.append(sigma_boot)
        except Exception:
            continue

    selected_sigmas = np.array(selected_sigmas)

    if len(selected_sigmas) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "all_selections": np.array([]),
        }

    return {
        "mean": np.mean(selected_sigmas),
        "std": np.std(selected_sigmas),
        "median": np.median(selected_sigmas),
        "q25": np.percentile(selected_sigmas, 25),
        "q75": np.percentile(selected_sigmas, 75),
        "coefficient_of_variation": np.std(selected_sigmas) / np.mean(selected_sigmas),
        "all_selections": selected_sigmas,
    }


# Convenience functions for integration with existing code
def select_rbf_bandwidth_cv_simple(
    X: NDArray[np.floating],
    cv_folds: int = 5,
    method: str = "kfold",
    scoring: str = "likelihood",
    n_candidates: int = 20,
    sigma_range: Optional[Tuple[float, float]] = None,
) -> float:
    """
    Simplified interface for bandwidth selection.

    Parameters
    ----------
    X : NDArray
        Input data of shape (n, d)
    cv_folds : int
        Number of CV folds
    method : str
        CV method ('kfold', 'timeseries', 'loo')
    scoring : str
        Scoring method ('likelihood', 'changepoint')
    n_candidates : int
        Number of bandwidth candidates
    sigma_range : Optional[Tuple[float, float]]
        Search range (auto-determined if None)

    Returns
    -------
    float
        Optimal bandwidth
    """
    # Auto-determine range if not provided
    if sigma_range is None:
        median_dist = _median_heuristic(X)
        sigma_range = (median_dist * 0.1, median_dist * 10.0)

    config = BandwidthCVConfig(
        method=method,
        cv_folds=cv_folds,
        scoring=scoring,
        n_candidates=n_candidates,
        sigma_range=sigma_range,
        search_strategy="adaptive_grid",
    )

    return select_rbf_bandwidth_cv(X, config)


# Example usage and testing
if __name__ == "__main__":
    print("Testing bandwidth cross-validation...")

    # Generate test data with multiple scales
    np.random.seed(42)
    n, d = 500, 3

    # Create data with structure at different scales
    # Fine scale: tight clusters
    X1 = np.random.normal([0, 0, 0], 0.5, size=(n // 3, d))
    X2 = np.random.normal([3, 3, 3], 0.5, size=(n // 3, d))

    # Coarse scale: spread out
    X3 = np.random.normal([0, 3, -3], 2.0, size=(n // 3, d))

    X = np.vstack([X1, X2, X3])

    print(f"Generated {n} samples in {d} dimensions")

    # Test different selection methods
    print("\n1. Cross-validation bandwidth selection:")
    sigma_cv = select_rbf_bandwidth_cv_simple(X, cv_folds=5)
    print(f"   CV selected σ = {sigma_cv:.4f}")

    print("\n2. Information criterion bandwidth selection:")
    sigma_bic = select_rbf_bandwidth_information_criterion(X, criterion="bic")
    print(f"   BIC selected σ = {sigma_bic:.4f}")

    sigma_aic = select_rbf_bandwidth_information_criterion(X, criterion="aic")
    print(f"   AIC selected σ = {sigma_aic:.4f}")

    print("\n3. Multi-scale bandwidth selection:")
    multiscale_results = select_rbf_bandwidth_multiscale(X, n_scales=3)
    for scale, sigma in multiscale_results.items():
        print(f"   {scale}: σ = {sigma:.4f}")

    print("\n4. Bandwidth stability analysis:")
    stability = bandwidth_stability_analysis(X, n_bootstrap=20)
    print(f"   Mean: {stability['mean']:.4f} ± {stability['std']:.4f}")
    print(f"   Median: {stability['median']:.4f}")
    print(f"   IQR: [{stability['q25']:.4f}, {stability['q75']:.4f}]")
    print(f"   Coefficient of variation: {stability['coefficient_of_variation']:.3f}")

    # Compare with median heuristic baseline
    sigma_median = _median_heuristic(X)
    print(f"\n5. Median heuristic baseline: σ = {sigma_median:.4f}")

    print("\nComparison:")
    print(f"   CV/Median ratio: {sigma_cv / sigma_median:.2f}")
    print(f"   BIC/Median ratio: {sigma_bic / sigma_median:.2f}")
    print(f"   AIC/Median ratio: {sigma_aic / sigma_median:.2f}")
