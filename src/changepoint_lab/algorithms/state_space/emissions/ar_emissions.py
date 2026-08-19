# ar_emissions.py
# MIT License
"""
Autoregressive (AR) emission distributions for HMM/HSMM.
Supports vector AR(p) processes with state-dependent parameters.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ARParams:
    """
    Autoregressive model parameters for K states.

    For each state k, the AR(p) model is:
    X_t = c_k + sum_{i=1}^p A_{k,i} * X_{t-i} + epsilon_t
    where epsilon_t ~ N(0, Sigma_k)

    Attributes
    ----------
    intercepts : NDArray[np.floating]
        Shape (K, D) - intercept vectors c_k for each state
    coeffs : NDArray[np.floating]
        Shape (K, p, D, D) - AR coefficient matrices A_{k,i}
    noise_covs : NDArray[np.floating]
        Shape (K, D, D) - noise covariance matrices Sigma_k
    order : int
        AR order p
    """

    intercepts: NDArray[np.floating]  # (K, D)
    coeffs: NDArray[np.floating]  # (K, p, D, D)
    noise_covs: NDArray[np.floating]  # (K, D, D)
    order: int

    def __post_init__(self):
        K, D = self.intercepts.shape
        p = self.order
        assert self.coeffs.shape == (K, p, D, D), f"Expected coeffs shape ({K}, {p}, {D}, {D})"
        assert self.noise_covs.shape == (K, D, D), f"Expected noise_covs shape ({K}, {D}, {D})"


def _build_design_matrix(X: NDArray[np.floating], order: int) -> NDArray[np.floating]:
    """
    Build design matrix for AR(p) regression from time series data.

    Parameters
    ----------
    X : NDArray
        Shape (T, D) - time series observations
    order : int
        AR order p

    Returns
    -------
    design : NDArray
        Shape (T-p, p*D + 1) - design matrix [1, X_{t-1}, ..., X_{t-p}]
        First column is ones (for intercept), remaining are lagged observations
    """
    T, D = X.shape
    if T <= order:
        raise ValueError(f"Need at least {order + 1} observations for AR({order})")

    T_effective = T - order
    design = np.ones((T_effective, order * D + 1), dtype=float)

    # Fill lagged observations
    for lag in range(1, order + 1):
        start_col = 1 + (lag - 1) * D
        end_col = start_col + D
        design[:, start_col:end_col] = X[order - lag : T - lag]

    return design


def _ar_log_likelihood_single(X: NDArray[np.floating], params: ARParams, state: int) -> float:
    """
    Compute log-likelihood for AR model in a single state.

    Parameters
    ----------
    X : NDArray
        Shape (T, D) - observations
    params : ARParams
        AR parameters
    state : int
        State index

    Returns
    -------
    loglik : float
        Log-likelihood for this state
    """
    T, D = X.shape
    p = params.order

    if T <= p:
        return -np.inf

    # Extract parameters for this state
    c_k = params.intercepts[state]  # (D,)
    A_k = params.coeffs[state]  # (p, D, D)
    Sigma_k = params.noise_covs[state]  # (D, D)

    # Compute residuals
    residuals = np.zeros((T - p, D), dtype=float)

    for t in range(p, T):
        # Predicted mean: c + sum_i A_i * X_{t-i}
        pred_mean = c_k.copy()
        for lag in range(1, p + 1):
            pred_mean += A_k[lag - 1] @ X[t - lag]

        residuals[t - p] = X[t] - pred_mean

    # Compute log-likelihood using multivariate normal
    try:
        # Cholesky decomposition for efficiency
        L = np.linalg.cholesky(Sigma_k)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))

        # Solve L @ L.T @ inv_Sigma = I for inv_Sigma
        I = np.eye(D)
        Y = np.linalg.solve(L, I)
        inv_Sigma = np.linalg.solve(L.T, Y)

    except np.linalg.LinAlgError:
        # Fallback to SVD
        U, s, Vt = np.linalg.svd(Sigma_k)
        s = np.maximum(s, 1e-12)
        log_det = np.sum(np.log(s))
        inv_Sigma = U @ np.diag(1.0 / s) @ Vt

    # Compute Mahalanobis distances
    loglik = 0.0
    const = -0.5 * (D * np.log(2 * np.pi) + log_det)

    for t in range(T - p):
        resid_t = residuals[t]
        mahal_dist = resid_t @ inv_Sigma @ resid_t
        loglik += const - 0.5 * mahal_dist

    return loglik


def ar_loglik(X: NDArray[np.floating], params: ARParams) -> NDArray[np.floating]:
    """
    Compute log-likelihood matrix for AR emissions.

    Parameters
    ----------
    X : NDArray
        Shape (T, D) - observations
    params : ARParams
        AR parameters for K states

    Returns
    -------
    loglik : NDArray
        Shape (T, K) - log P(X_t | state=k)
        Note: First p time points get -inf likelihood due to AR lag
    """
    T, D = X.shape
    K = params.intercepts.shape[0]
    p = params.order

    loglik = np.full((T, K), -np.inf, dtype=float)

    if T <= p:
        return loglik

    # For AR models, we can only compute likelihood starting from time p
    # Each observation X_t depends on X_{t-1}, ..., X_{t-p}
    for k in range(K):
        c_k = params.intercepts[k]
        A_k = params.coeffs[k]
        Sigma_k = params.noise_covs[k]

        # Ensure covariance is PSD
        try:
            L = np.linalg.cholesky(Sigma_k)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            I = np.eye(D)
            Y = np.linalg.solve(L, I)
            inv_Sigma = np.linalg.solve(L.T, Y)
        except np.linalg.LinAlgError:
            U, s, Vt = np.linalg.svd(Sigma_k)
            s = np.maximum(s, 1e-12)
            log_det = np.sum(np.log(s))
            inv_Sigma = U @ np.diag(1.0 / s) @ Vt

        const = -0.5 * (D * np.log(2 * np.pi) + log_det)

        # Compute likelihood for each time point t >= p
        for t in range(p, T):
            # Predicted mean
            pred_mean = c_k.copy()
            for lag in range(1, p + 1):
                pred_mean += A_k[lag - 1] @ X[t - lag]

            # Residual and Mahalanobis distance
            residual = X[t] - pred_mean
            mahal_dist = residual @ inv_Sigma @ residual

            loglik[t, k] = const - 0.5 * mahal_dist

    return loglik


def estimate_ar_from_labels(
    X: NDArray[np.floating], labels: NDArray[np.integer], K: int, order: int, reg: float = 1e-6
) -> ARParams:
    """
    Estimate AR parameters from hard state labels using least squares.

    Parameters
    ----------
    X : NDArray
        Shape (T, D) - observations
    labels : NDArray
        Shape (T,) - hard state assignments
    K : int
        Number of states
    order : int
        AR order
    reg : float
        Regularization for covariance estimation

    Returns
    -------
    ARParams
        Estimated parameters
    """
    T, D = X.shape

    intercepts = np.zeros((K, D), dtype=float)
    coeffs = np.zeros((K, order, D, D), dtype=float)
    noise_covs = np.zeros((K, D, D), dtype=float)

    for k in range(K):
        # Find segments for this state
        state_mask = labels == k
        state_indices = np.where(state_mask)[0]

        if len(state_indices) <= order:
            # Not enough data - use global statistics
            intercepts[k] = X.mean(axis=0)
            noise_covs[k] = np.cov(X, rowvar=False) + reg * np.eye(D)
            continue

        # Build design matrix and response from state segments
        # Need to be careful about temporal continuity within states
        design_matrices = []
        responses = []

        # Group consecutive indices
        groups = []
        current_group = [state_indices[0]]

        for i in range(1, len(state_indices)):
            if state_indices[i] == state_indices[i - 1] + 1:
                current_group.append(state_indices[i])
            else:
                if len(current_group) > order:
                    groups.append(current_group)
                current_group = [state_indices[i]]

        if len(current_group) > order:
            groups.append(current_group)

        # Extract data from valid groups
        for group in groups:
            group_X = X[group]  # Shape (len(group), D)
            if len(group) > order:
                design = _build_design_matrix(group_X, order)  # (len(group)-p, p*D+1)
                response = group_X[order:]  # (len(group)-p, D)

                design_matrices.append(design)
                responses.append(response)

        if not design_matrices:
            # No valid segments - use global mean and identity
            intercepts[k] = X.mean(axis=0)
            noise_covs[k] = np.cov(X, rowvar=False) + reg * np.eye(D)
            continue

        # Combine all design matrices and responses
        full_design = np.vstack(design_matrices)  # (N_total, p*D+1)
        full_response = np.vstack(responses)  # (N_total, D)

        # Solve least squares for each output dimension
        for d in range(D):
            y_d = full_response[:, d]  # (N_total,)

            try:
                # Solve normal equations: (X^T X) beta = X^T y
                beta_d = np.linalg.solve(
                    full_design.T @ full_design + reg * np.eye(full_design.shape[1]),
                    full_design.T @ y_d,
                )
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse
                beta_d = np.linalg.pinv(full_design) @ y_d

            # Extract intercept and coefficients
            intercepts[k, d] = beta_d[0]

            # Reshape coefficient vector back to matrices
            for lag in range(order):
                start_idx = 1 + lag * D
                end_idx = start_idx + D
                coeffs[k, lag, d, :] = beta_d[start_idx:end_idx]

        # Estimate noise covariance from residuals
        pred_response = (
            full_design
            @ np.column_stack(
                [np.concatenate([[intercepts[k, d]], coeffs[k, :, d, :].ravel()]) for d in range(D)]
            )
        )  # (N_total, D)

        residuals = full_response - pred_response

        if residuals.shape[0] > 1:
            noise_covs[k] = np.cov(residuals, rowvar=False) + reg * np.eye(D)
        else:
            noise_covs[k] = reg * np.eye(D)

        # Ensure covariance is well-conditioned
        if noise_covs[k].ndim == 0:
            noise_covs[k] = np.array([[noise_covs[k] + reg]], dtype=float)

    return ARParams(intercepts=intercepts, coeffs=coeffs, noise_covs=noise_covs, order=order)


def estimate_ar_from_responsibilities(
    X: NDArray[np.floating], responsibilities: NDArray[np.floating], order: int, reg: float = 1e-6
) -> ARParams:
    """
    Estimate AR parameters from soft responsibilities using weighted least squares.

    Parameters
    ----------
    X : NDArray
        Shape (T, D) - observations
    responsibilities : NDArray
        Shape (T, K) - soft state assignments P(state=k | X_t)
    order : int
        AR order
    reg : float
        Regularization parameter

    Returns
    -------
    ARParams
        Estimated parameters
    """
    T, D = X.shape
    K = responsibilities.shape[1]

    if T <= order:
        raise ValueError(f"Need at least {order + 1} time points for AR({order})")

    intercepts = np.zeros((K, D), dtype=float)
    coeffs = np.zeros((K, order, D, D), dtype=float)
    noise_covs = np.zeros((K, D, D), dtype=float)

    # Build design matrix for all valid time points
    design = _build_design_matrix(X, order)  # (T-p, p*D+1)
    response = X[order:]  # (T-p, D)
    weights = responsibilities[order:]  # (T-p, K)

    for k in range(K):
        w_k = weights[:, k]  # (T-p,)
        total_weight = w_k.sum()

        if total_weight < 1e-12:
            # No effective observations for this state
            intercepts[k] = X.mean(axis=0)
            noise_covs[k] = np.cov(X, rowvar=False) + reg * np.eye(D)
            continue

        # Weighted design matrix and response
        sqrt_w_k = np.sqrt(w_k)  # (T-p,)
        weighted_design = design * sqrt_w_k[:, None]  # (T-p, p*D+1)
        weighted_response = response * sqrt_w_k[:, None]  # (T-p, D)

        # Solve weighted least squares for each output dimension
        for d in range(D):
            y_d = weighted_response[:, d]  # (T-p,)

            try:
                # Solve (X^T W X + λI) β = X^T W y
                XTX = weighted_design.T @ weighted_design
                XTy = weighted_design.T @ y_d
                beta_d = np.linalg.solve(XTX + reg * np.eye(XTX.shape[0]), XTy)
            except np.linalg.LinAlgError:
                beta_d = np.linalg.pinv(weighted_design) @ y_d

            # Extract parameters
            intercepts[k, d] = beta_d[0]

            for lag in range(order):
                start_idx = 1 + lag * D
                end_idx = start_idx + D
                coeffs[k, lag, d, :] = beta_d[start_idx:end_idx]

        # Estimate weighted noise covariance
        # Predict responses
        pred_response = np.zeros_like(response)
        for t in range(response.shape[0]):
            pred_response[t] = intercepts[k].copy()
            for lag in range(order):
                pred_response[t] += coeffs[k, lag] @ X[order + t - lag - 1]

        residuals = response - pred_response  # (T-p, D)

        # Weighted covariance: Cov = (1/W) * sum_t w_t * r_t * r_t^T
        noise_cov_k = np.zeros((D, D), dtype=float)
        for t in range(residuals.shape[0]):
            if w_k[t] > 1e-12:
                outer = np.outer(residuals[t], residuals[t])
                noise_cov_k += w_k[t] * outer

        noise_cov_k = noise_cov_k / total_weight + reg * np.eye(D)
        noise_covs[k] = noise_cov_k

    return ARParams(intercepts=intercepts, coeffs=coeffs, noise_covs=noise_covs, order=order)


def simulate_ar_process(
    params: ARParams,
    state_sequence: NDArray[np.integer],
    X_init: Optional[NDArray[np.floating]] = None,
    seed: Optional[int] = None,
) -> NDArray[np.floating]:
    """
    Simulate AR time series given state sequence.

    Parameters
    ----------
    params : ARParams
        AR parameters
    state_sequence : NDArray
        Shape (T,) - state sequence
    X_init : Optional[NDArray]
        Shape (p, D) - initial values. If None, use zeros.
    seed : Optional[int]
        Random seed

    Returns
    -------
    X : NDArray
        Shape (T, D) - simulated time series
    """
    rng = np.random.default_rng(seed)
    T = len(state_sequence)
    D = params.intercepts.shape[1]
    p = params.order

    # Initialize output
    X = np.zeros((T, D), dtype=float)

    # Set initial values
    if X_init is not None:
        assert X_init.shape == (p, D), f"X_init must have shape ({p}, {D})"
        if T >= p:
            X[:p] = X_init
    else:
        # Initialize with small random values
        if T >= p:
            X[:p] = rng.normal(0, 0.1, size=(p, D))

    # Generate time series
    for t in range(p, T):
        k = state_sequence[t]

        # Compute mean prediction
        pred_mean = params.intercepts[k].copy()
        for lag in range(1, p + 1):
            pred_mean += params.coeffs[k, lag - 1] @ X[t - lag]

        # Add noise
        noise = rng.multivariate_normal(np.zeros(D), params.noise_covs[k])

        X[t] = pred_mean + noise

    return X


class AREmissions:
    """
    Autoregressive emission model for HMM/HSMM.

    Compatible interface with the HSMM framework.
    """

    def __init__(self, n_states: int, order: int = 1, reg: float = 1e-6):
        """
        Parameters
        ----------
        n_states : int
            Number of hidden states
        order : int
            AR order (default: AR(1))
        reg : float
            Regularization parameter
        """
        self.n_states = n_states
        self.order = order
        self.reg = reg
        self.params: Optional[ARParams] = None

    def initialize(
        self, X: NDArray[np.floating], method: str = "random", seed: Optional[int] = None
    ) -> None:
        """
        Initialize AR parameters.

        Parameters
        ----------
        X : NDArray
            Shape (T, D) - training data for initialization
        method : str
            Initialization method: 'random', 'global', 'kmeans'
        seed : Optional[int]
            Random seed
        """
        rng = np.random.default_rng(seed)
        T, D = X.shape

        if method == "random":
            self._initialize_random(X, rng)
        elif method == "global":
            self._initialize_global(X, rng)
        elif method == "kmeans":
            self._initialize_kmeans(X, rng)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def _initialize_random(
        self,
        X: NDArray[np.floating],
        rng: np.random.Generator,
    ) -> None:
        """Initialize parameters randomly around data statistics."""
        T, D = X.shape

        # Random intercepts around data mean
        data_mean = X.mean(axis=0)
        data_std = X.std(axis=0)

        intercepts = np.zeros((self.n_states, D), dtype=float)
        for k in range(self.n_states):
            intercepts[k] = data_mean + rng.normal(0, data_std * 0.5)

        # Random AR coefficients (small values for stability)
        coeffs = rng.normal(0, 0.1, size=(self.n_states, self.order, D, D))

        # Diagonal structure for stability
        for k in range(self.n_states):
            for lag in range(self.order):
                coeffs[k, lag] = np.diag(np.diag(coeffs[k, lag]))

        # Noise covariances around empirical covariance
        emp_cov = np.cov(X, rowvar=False)
        if emp_cov.ndim == 0:
            emp_cov = np.array([[emp_cov]], dtype=float)

        noise_covs = np.zeros((self.n_states, D, D), dtype=float)
        for k in range(self.n_states):
            scale = rng.uniform(0.5, 2.0)
            noise_covs[k] = scale * emp_cov + self.reg * np.eye(D)

        self.params = ARParams(
            intercepts=intercepts, coeffs=coeffs, noise_covs=noise_covs, order=self.order
        )

    def _initialize_global(
        self,
        X: NDArray[np.floating],
        rng: np.random.Generator,
    ) -> None:
        """Initialize all states with global AR parameters."""
        # Fit single AR model to entire data
        dummy_labels = np.zeros(X.shape[0], dtype=int)
        global_params = estimate_ar_from_labels(X, dummy_labels, 1, self.order, self.reg)

        # Replicate for all states with small perturbations
        intercepts = np.tile(global_params.intercepts[0], (self.n_states, 1))
        coeffs = np.tile(global_params.coeffs[0], (self.n_states, 1, 1, 1))
        noise_covs = np.tile(global_params.noise_covs[0], (self.n_states, 1, 1))

        # Add small random perturbations
        for k in range(1, self.n_states):
            intercepts[k] += rng.normal(0, 0.1, size=intercepts.shape[1])
            coeffs[k] += rng.normal(0, 0.05, size=coeffs.shape[1:])

        self.params = ARParams(
            intercepts=intercepts, coeffs=coeffs, noise_covs=noise_covs, order=self.order
        )

    def _initialize_kmeans(
        self,
        X: NDArray[np.floating],
        rng: np.random.Generator,
    ) -> None:
        """Initialize using k-means clustering and fit AR to each cluster."""
        # Simple k-means (could use more sophisticated version)
        T, D = X.shape

        # Use every 'order+1' points to avoid temporal dependencies in clustering
        subsample_idx = np.arange(self.order, T, self.order + 1)
        X_sub = X[subsample_idx]

        if len(X_sub) < self.n_states:
            # Fallback to random initialization
            self._initialize_random(X, rng)
            return

        # K-means clustering
        centers = X_sub[rng.choice(len(X_sub), self.n_states, replace=False)]

        for _ in range(20):  # Max iterations
            # Assignment step
            distances = np.sum((X_sub[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update step
            new_centers = np.zeros_like(centers)
            for k in range(self.n_states):
                mask = labels == k
                if np.any(mask):
                    new_centers[k] = X_sub[mask].mean(axis=0)
                else:
                    new_centers[k] = centers[k]

            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        # Expand labels to full time series (simple nearest neighbor)
        full_labels = np.zeros(T, dtype=int)
        for t in range(T):
            dists = np.sum((X[t] - centers) ** 2, axis=1)
            full_labels[t] = np.argmin(dists)

        # Fit AR parameters
        self.params = estimate_ar_from_labels(X, full_labels, self.n_states, self.order, self.reg)

    def compute_loglik(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute log-likelihood matrix."""
        if self.params is None:
            raise RuntimeError("Parameters not initialized. Call initialize() first.")

        return ar_loglik(X, self.params)

    def update_from_responsibilities(
        self, X: NDArray[np.floating], responsibilities: NDArray[np.floating]
    ) -> None:
        """Update parameters from soft responsibilities (M-step)."""
        self.params = estimate_ar_from_responsibilities(X, responsibilities, self.order, self.reg)

    def sample(
        self,
        states: NDArray[np.integer],
        X_init: Optional[NDArray[np.floating]] = None,
        seed: Optional[int] = None,
    ) -> NDArray[np.floating]:
        """Sample observations given state sequence."""
        if self.params is None:
            raise RuntimeError("Parameters not initialized.")

        return simulate_ar_process(self.params, states, X_init, seed)

    @property
    def n_parameters(self) -> int:
        """Number of free parameters."""
        if self.params is None:
            return 0

        D = self.params.intercepts.shape[1]
        # K intercepts (K*D) + K AR coefficient matrices (K*p*D*D) + K noise covariances (K*D*(D+1)/2)
        return self.n_states * (D + self.order * D * D + D * (D + 1) // 2)


# Example usage and testing
if __name__ == "__main__":
    print("Testing AR emissions...")

    # Generate synthetic AR(2) data with 2 states
    rng = np.random.default_rng(42)

    T, D, K, p = 1000, 2, 2, 2

    # True AR parameters
    true_intercepts = np.array([[0.0, 0.5], [1.0, -0.3]], dtype=float)

    # AR coefficients (stable dynamics)
    true_coeffs = np.zeros((K, p, D, D), dtype=float)
    true_coeffs[0, 0] = np.array([[0.7, 0.1], [0.0, 0.6]])  # lag-1 for state 0
    true_coeffs[0, 1] = np.array([[-0.2, 0.0], [0.1, -0.1]])  # lag-2 for state 0
    true_coeffs[1, 0] = np.array([[0.5, -0.1], [0.2, 0.8]])  # lag-1 for state 1
    true_coeffs[1, 1] = np.array([[0.1, 0.05], [-0.05, 0.0]])  # lag-2 for state 1

    # Noise covariances
    true_noise_covs = np.array(
        [
            [[1.0, 0.2], [0.2, 1.0]],  # state 0
            [[0.5, -0.1], [-0.1, 0.8]],  # state 1
        ],
        dtype=float,
    )

    true_params = ARParams(
        intercepts=true_intercepts, coeffs=true_coeffs, noise_covs=true_noise_covs, order=p
    )

    # Generate state sequence (simple alternating)
    states = rng.choice(K, size=T)

    # Generate AR time series
    X = simulate_ar_process(true_params, states, seed=42)

    print(f"Generated {T} samples of {D}-dimensional AR({p}) data with {K} states")

    # Test AR emissions class
    ar_emissions = AREmissions(n_states=K, order=p)

    # Test different initialization methods
    for method in ["random", "global", "kmeans"]:
        print(f"\nTesting {method} initialization:")
        ar_emissions.initialize(X, method=method, seed=42)

        loglik = ar_emissions.compute_loglik(X)
        print(f"  Log-likelihood shape: {loglik.shape}")
        print(f"  Mean log-likelihood: {loglik[p:].mean():.4f}")  # Skip first p points
        print(f"  Number of parameters: {ar_emissions.n_parameters}")

    # Test parameter estimation from labels
    print("\nTesting parameter estimation from true labels:")
    true_est_params = estimate_ar_from_labels(X, states, K, p)

    print(f"True intercepts:\n{true_intercepts}")
    print(f"Estimated intercepts:\n{true_est_params.intercepts}")

    intercept_error = np.mean((true_intercepts - true_est_params.intercepts) ** 2)
    print(f"Intercept MSE: {intercept_error:.6f}")

    # Test soft parameter estimation (simulate soft labels)
    loglik = ar_loglik(X, true_est_params)
    # Convert to responsibilities (softmax)
    max_loglik = np.max(loglik, axis=1, keepdims=True)
    exp_loglik = np.exp(loglik - max_loglik)
    responsibilities = exp_loglik / np.sum(exp_loglik, axis=1, keepdims=True)

    soft_params = estimate_ar_from_responsibilities(X, responsibilities, p)
    ar_emissions.params = soft_params
    new_loglik = ar_emissions.compute_loglik(X)

    print("\nAfter soft parameter update:")
    print(f"  Improved mean log-likelihood: {new_loglik[p:].mean():.4f}")
