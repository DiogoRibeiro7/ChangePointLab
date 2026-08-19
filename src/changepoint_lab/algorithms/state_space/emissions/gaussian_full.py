# gaussian_full.py
# MIT License
"""
Full-covariance Gaussian emissions for HMM/HSMM with numerically stable implementation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GaussianFullParams:
    """
    Full-covariance Gaussian parameters for K states.

    Attributes
    ----------
    means : NDArray[np.floating]
        Shape (K, D) - mean vectors for each state
    covs : NDArray[np.floating]
        Shape (K, D, D) - covariance matrices for each state
    """

    means: NDArray[np.floating]  # (K, D)
    covs: NDArray[np.floating]  # (K, D, D)

    def __post_init__(self):
        K, D = self.means.shape
        assert self.covs.shape == (K, D, D), (
            f"Expected covs shape ({K}, {D}, {D}), got {self.covs.shape}"
        )


def _ensure_psd(cov: NDArray[np.floating], eps: float = 1e-6) -> NDArray[np.floating]:
    """
    Ensure covariance matrix is positive semi-definite via eigenvalue clamping.
    """
    # Symmetrize
    cov = 0.5 * (cov + cov.T)

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Clamp eigenvalues
    eigvals = np.maximum(eigvals, eps)

    # Reconstruct
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _stable_logdet_inv(cov: NDArray[np.floating]) -> Tuple[float, NDArray[np.floating]]:
    """
    Compute log determinant and inverse of covariance matrix stably via Cholesky.

    Returns
    -------
    logdet : float
        Log determinant of covariance matrix
    inv_cov : NDArray
        Inverse covariance matrix
    """
    try:
        # Try Cholesky decomposition (most efficient for PSD matrices)
        L = np.linalg.cholesky(cov)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))

        # Solve L @ L.T @ inv_cov = I via two triangular solves
        I = np.eye(cov.shape[0])
        Y = np.linalg.solve(L, I)
        inv_cov = np.linalg.solve(L.T, Y)

    except np.linalg.LinAlgError:
        # Fallback to SVD for numerical issues
        U, s, Vt = np.linalg.svd(cov)
        s = np.maximum(s, 1e-12)  # Clamp small singular values
        logdet = np.sum(np.log(s))
        inv_cov = U @ np.diag(1.0 / s) @ Vt

    return logdet, inv_cov


def gaussian_full_loglik(
    X: NDArray[np.floating], params: GaussianFullParams, allow_nan: bool = False
) -> NDArray[np.floating]:
    """
    Compute log-likelihood matrix for full-covariance Gaussians.

    Parameters
    ----------
    X : NDArray
        Shape (T, D) - observations
    params : GaussianFullParams
        Gaussian parameters for K states
    allow_nan : bool
        If True, NaN observations get -inf likelihood

    Returns
    -------
    loglik : NDArray
        Shape (T, K) - log P(X_t | state=k)
    """
    T, D = X.shape
    K = params.means.shape[0]

    loglik = np.full((T, K), -np.inf, dtype=float)

    # Precompute inverse covariances and log determinants
    inv_covs = np.zeros_like(params.covs)
    logdets = np.zeros(K, dtype=float)

    for k in range(K):
        cov_k = _ensure_psd(params.covs[k])
        logdets[k], inv_covs[k] = _stable_logdet_inv(cov_k)

    # Normalization constant
    log_norm = -0.5 * (D * np.log(2 * np.pi) + logdets)  # (K,)

    # Compute likelihoods
    for t in range(T):
        x_t = X[t]

        if allow_nan and np.any(np.isnan(x_t)):
            continue  # loglik[t, :] remains -inf

        for k in range(K):
            diff = x_t - params.means[k]  # (D,)
            mahal = diff @ inv_covs[k] @ diff  # Mahalanobis distance squared
            loglik[t, k] = log_norm[k] - 0.5 * mahal

    return loglik


def estimate_gaussian_full_from_labels(
    X: NDArray[np.floating], labels: NDArray[np.integer], K: int, reg: float = 1e-6
) -> GaussianFullParams:
    """
    Estimate full-covariance Gaussian parameters from hard labels.

    Parameters
    ----------
    X : NDArray
        Shape (T, D) - observations
    labels : NDArray
        Shape (T,) - hard state assignments in {0, ..., K-1}
    K : int
        Number of states
    reg : float
        Regularization added to diagonal for numerical stability

    Returns
    -------
    GaussianFullParams
        Estimated parameters
    """
    T, D = X.shape

    means = np.zeros((K, D), dtype=float)
    covs = np.zeros((K, D, D), dtype=float)

    for k in range(K):
        mask = labels == k
        X_k = X[mask]

        if len(X_k) == 0:
            # No observations for this state - use global statistics
            means[k] = X.mean(axis=0)
            covs[k] = np.cov(X, rowvar=False) + reg * np.eye(D)
        elif len(X_k) == 1:
            # Single observation - use mean and regularized identity
            means[k] = X_k[0]
            covs[k] = reg * np.eye(D)
        else:
            # Multiple observations
            means[k] = X_k.mean(axis=0)
            cov_k = np.cov(X_k, rowvar=False, bias=False)

            # Handle scalar case
            if cov_k.ndim == 0:
                cov_k = np.array([[cov_k]], dtype=float)

            # Regularize
            covs[k] = cov_k + reg * np.eye(D)

    return GaussianFullParams(means=means, covs=covs)


def estimate_gaussian_full_from_responsibilities(
    X: NDArray[np.floating], responsibilities: NDArray[np.floating], reg: float = 1e-6
) -> GaussianFullParams:
    """
    Estimate full-covariance Gaussian parameters from soft responsibilities (E-step output).

    Parameters
    ----------
    X : NDArray
        Shape (T, D) - observations
    responsibilities : NDArray
        Shape (T, K) - posterior probabilities P(state=k | X_t)
    reg : float
        Regularization for covariance matrices

    Returns
    -------
    GaussianFullParams
        Estimated parameters
    """
    T, D = X.shape
    K = responsibilities.shape[1]

    # Effective counts
    N_k = responsibilities.sum(axis=0)  # (K,)
    N_k = np.maximum(N_k, 1e-12)  # Avoid division by zero

    # Weighted means
    means = np.zeros((K, D), dtype=float)
    for k in range(K):
        means[k] = (responsibilities[:, k : k + 1].T @ X) / N_k[k]  # (1, D) -> (D,)
        means[k] = means[k].ravel()

    # Weighted covariances
    covs = np.zeros((K, D, D), dtype=float)
    for k in range(K):
        # Center observations
        X_centered = X - means[k]  # (T, D)

        # Weighted outer products
        weighted_outer = np.zeros((D, D), dtype=float)
        for t in range(T):
            w_t = responsibilities[t, k]
            if w_t > 1e-12:
                outer_t = np.outer(X_centered[t], X_centered[t])
                weighted_outer += w_t * outer_t

        # Normalize and regularize
        covs[k] = weighted_outer / N_k[k] + reg * np.eye(D)

    return GaussianFullParams(means=means, covs=covs)


def estimate_gaussian_full_by_kmeans(
    X: NDArray[np.floating],
    K: int,
    n_init: int = 5,
    max_iter: int = 100,
    reg: float = 1e-6,
    seed: Optional[int] = None,
) -> GaussianFullParams:
    """
    Initialize full-covariance Gaussians using k-means++ followed by hard EM.

    Parameters
    ----------
    X : NDArray
        Shape (T, D) - observations
    K : int
        Number of components
    n_init : int
        Number of random initializations
    max_iter : int
        Maximum k-means iterations per initialization
    reg : float
        Covariance regularization
    seed : Optional[int]
        Random seed

    Returns
    -------
    GaussianFullParams
        Estimated parameters from best k-means run
    """
    rng = np.random.default_rng(seed)
    T, D = X.shape
    best_inertia = np.inf
    best_params = None

    for _ in range(n_init):
        # k-means++ initialization
        centers = np.zeros((K, D), dtype=float)

        # Choose first center randomly
        centers[0] = X[int(rng.integers(T))]

        # Choose remaining centers with probability proportional to squared distance
        for k in range(1, K):
            # Compute distances to nearest existing center
            min_dists = np.full(T, np.inf, dtype=float)
            for j in range(k):
                dists = np.sum((X - centers[j]) ** 2, axis=1)
                min_dists = np.minimum(min_dists, dists)

            # Sample proportional to squared distances
            probs = min_dists / min_dists.sum()
            centers[k] = X[int(rng.choice(T, p=probs))]

        # Run k-means
        for _ in range(max_iter):
            # Assignment step
            distances = np.zeros((T, K), dtype=float)
            for k in range(K):
                distances[:, k] = np.sum((X - centers[k]) ** 2, axis=1)

            labels = np.argmin(distances, axis=1)

            # Update step
            new_centers = np.zeros_like(centers)
            for k in range(K):
                mask = labels == k
                if np.any(mask):
                    new_centers[k] = X[mask].mean(axis=0)
                else:
                    new_centers[k] = centers[k]  # Keep old center if no assignments

            # Check convergence
            if np.allclose(centers, new_centers):
                break

            centers = new_centers

        # Compute inertia (within-cluster sum of squares)
        inertia = 0.0
        for k in range(K):
            mask = labels == k
            if np.any(mask):
                inertia += np.sum((X[mask] - centers[k]) ** 2)

        # Keep best result
        if inertia < best_inertia:
            best_inertia = inertia
            best_params = estimate_gaussian_full_from_labels(X, labels, K, reg=reg)

    return best_params


class GaussianFullEmissions:
    """
    Full-covariance Gaussian emission model for HMM/HSMM.

    Provides a unified interface for initialization, parameter estimation,
    and likelihood computation compatible with the HSMM framework.
    """

    def __init__(self, n_states: int, reg: float = 1e-6):
        """
        Parameters
        ----------
        n_states : int
            Number of hidden states
        reg : float
            Regularization parameter for covariance matrices
        """
        self.n_states = n_states
        self.reg = reg
        self.params: Optional[GaussianFullParams] = None

    def initialize_kmeans(
        self, X: NDArray[np.floating], n_init: int = 5, seed: Optional[int] = None
    ) -> None:
        """Initialize parameters using k-means++."""
        self.params = estimate_gaussian_full_by_kmeans(
            X, self.n_states, n_init=n_init, reg=self.reg, seed=seed
        )

    def initialize_random(self, X: NDArray[np.floating], seed: Optional[int] = None) -> None:
        """Initialize parameters randomly around data statistics."""
        rng = np.random.default_rng(seed)

        T, D = X.shape

        # Initialize means as random data points with noise
        data_mean = X.mean(axis=0)
        data_std = X.std(axis=0)

        means = np.zeros((self.n_states, D), dtype=float)
        for k in range(self.n_states):
            means[k] = data_mean + rng.normal(0, data_std * 0.5, size=D)

        # Initialize covariances as scaled identity around empirical covariance
        emp_cov = np.cov(X, rowvar=False)
        if emp_cov.ndim == 0:
            emp_cov = np.array([[emp_cov]], dtype=float)

        covs = np.zeros((self.n_states, D, D), dtype=float)
        for k in range(self.n_states):
            scale = rng.uniform(0.5, 2.0)  # Random scaling
            covs[k] = scale * emp_cov + self.reg * np.eye(D)

        self.params = GaussianFullParams(means=means, covs=covs)

    def compute_loglik(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute log-likelihood matrix.

        Parameters
        ----------
        X : NDArray
            Shape (T, D) - observations

        Returns
        -------
        loglik : NDArray
            Shape (T, K) - log P(X_t | state=k)
        """
        if self.params is None:
            raise RuntimeError("Parameters not initialized. Call initialize_* first.")

        return gaussian_full_loglik(X, self.params)

    def update_from_responsibilities(
        self, X: NDArray[np.floating], responsibilities: NDArray[np.floating]
    ) -> None:
        """
        Update parameters from soft responsibilities (M-step).

        Parameters
        ----------
        X : NDArray
            Shape (T, D) - observations
        responsibilities : NDArray
            Shape (T, K) - posterior probabilities
        """
        self.params = estimate_gaussian_full_from_responsibilities(
            X, responsibilities, reg=self.reg
        )

    def sample(
        self, states: NDArray[np.integer], seed: Optional[int] = None
    ) -> NDArray[np.floating]:
        """
        Sample observations given state sequence.

        Parameters
        ----------
        states : NDArray
            Shape (T,) - state sequence
        seed : Optional[int]
            Random seed

        Returns
        -------
        X : NDArray
            Shape (T, D) - sampled observations
        """
        if self.params is None:
            raise RuntimeError("Parameters not initialized.")

        rng = np.random.default_rng(seed)
        T = len(states)
        D = self.params.means.shape[1]

        X = np.zeros((T, D), dtype=float)

        for t in range(T):
            k = states[t]
            mean_k = self.params.means[k]
            cov_k = self.params.covs[k]
            X[t] = rng.multivariate_normal(mean_k, cov_k)

        return X

    @property
    def n_parameters(self) -> int:
        """Number of free parameters in the model."""
        if self.params is None:
            return 0

        D = self.params.means.shape[1]
        # K mean vectors of dimension D + K covariance matrices of dimension D(D+1)/2
        return self.n_states * (D + D * (D + 1) // 2)


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic data
    rng = np.random.default_rng(42)

    T, D, K = 500, 3, 2

    # True parameters
    true_means = np.array([[0, 0, 0], [3, 3, 3]], dtype=float)
    true_covs = np.array(
        [
            [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]],
            [[2.0, -0.3, 0.1], [-0.3, 1.5, 0.0], [0.1, 0.0, 1.2]],
        ],
        dtype=float,
    )

    true_params = GaussianFullParams(means=true_means, covs=true_covs)

    # Generate data
    states = rng.choice(K, size=T)
    X = np.zeros((T, D), dtype=float)

    for t in range(T):
        k = states[t]
        X[t] = rng.multivariate_normal(true_means[k], true_covs[k])

    # Test estimation
    print("Testing full-covariance Gaussian emissions...")

    # Method 1: From hard labels
    est_params_hard = estimate_gaussian_full_from_labels(X, states, K)
    print("Hard labels estimation:")
    print(f"  Mean error: {np.mean((est_params_hard.means - true_means) ** 2):.4f}")

    # Method 2: k-means initialization
    emissions = GaussianFullEmissions(K)
    emissions.initialize_kmeans(X, n_init=3, seed=42)
    loglik = emissions.compute_loglik(X)
    print("k-means initialization:")
    print(f"  Log-likelihood shape: {loglik.shape}")
    print(f"  Mean log-likelihood: {loglik.mean():.4f}")

    # Method 3: From soft responsibilities (simulate E-step output)

    # responsibilities = softmax(loglik, axis=1)  # (T, K)
    # For NumPy-only version:
    exp_loglik = np.exp(loglik - np.max(loglik, axis=1, keepdims=True))
    responsibilities = exp_loglik / np.sum(exp_loglik, axis=1, keepdims=True)

    emissions.update_from_responsibilities(X, responsibilities)
    new_loglik = emissions.compute_loglik(X)
    print("After soft update:")
    print(f"  Improved log-likelihood: {new_loglik.mean():.4f}")

    print(f"Number of parameters: {emissions.n_parameters}")
