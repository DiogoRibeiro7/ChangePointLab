# utils.py
# MIT License
"""
Shared utilities for the Change-Point & State-Space Toolkit.
Provides common statistical, numerical, and matrix operations used across modules.
"""

from __future__ import annotations
from typing import Callable, List, Optional, Sequence, Tuple, Union, TypeVar

import numpy as np
from numpy.typing import NDArray


# ========================
# Statistical Functions
# ========================

def lgamma(x: float) -> float:
    """Natural log of the gamma function."""
    return float(np.log(np.abs(np.math.gamma(x))))


def log_beta(a: float, b: float) -> float:
    """
    Compute log Beta(a,b) = logGamma(a) + logGamma(b) - logGamma(a+b).
    
    Parameters
    ----------
    a, b : float
        Parameters of the beta function
    
    Returns
    -------
    float
        Natural logarithm of the beta function
    """
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def safe_normalize(x: NDArray[np.floating], axis: int = 0, eps: float = 1e-12) -> NDArray[np.floating]:
    """
    Normalize array to sum to 1 along the specified axis, with protection against division by zero.
    
    Parameters
    ----------
    x : NDArray
        Input array
    axis : int
        Axis along which to normalize
    eps : float
        Small constant to avoid division by zero
        
    Returns
    -------
    NDArray
        Normalized array
    """
    s = np.sum(x, axis=axis, keepdims=True)
    s = np.maximum(s, eps)
    return x / s


def softmax(x: NDArray[np.floating], axis: int = -1) -> NDArray[np.floating]:
    """
    Compute softmax values for each set of scores in x along specified axis.
    
    Parameters
    ----------
    x : NDArray
        Input array
    axis : int
        Axis along which to apply softmax
        
    Returns
    -------
    NDArray
        Softmax-transformed array
    """
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def logsumexp(x: NDArray[np.floating], axis: int = -1) -> NDArray[np.floating]:
    """
    Compute log(sum(exp(x))) in a numerically stable way.
    
    Parameters
    ----------
    x : NDArray
        Input array
    axis : int
        Axis along which to perform the operation
        
    Returns
    -------
    NDArray
        Result of the logsumexp operation
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    return x_max + np.log(np.sum(np.exp(x - x_max), axis=axis))


# ========================
# Matrix Operations
# ========================

def ensure_psd(cov: NDArray[np.floating], eps: float = 1e-6) -> NDArray[np.floating]:
    """
    Ensure covariance matrix is positive semi-definite via eigenvalue clamping.
    
    Parameters
    ----------
    cov : NDArray
        Input covariance matrix
    eps : float
        Minimum eigenvalue threshold
        
    Returns
    -------
    NDArray
        Positive semi-definite covariance matrix
    """
    # Symmetrize
    cov = 0.5 * (cov + cov.T)
    
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Clamp eigenvalues
    eigvals = np.maximum(eigvals, eps)
    
    # Reconstruct
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def stable_logdet_inv(cov: NDArray[np.floating]) -> Tuple[float, NDArray[np.floating]]:
    """
    Compute log determinant and inverse of covariance matrix stably via Cholesky decomposition.
    Falls back to SVD if Cholesky fails.
    
    Parameters
    ----------
    cov : NDArray
        Covariance matrix
        
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


def mahalanobis_distance(x: NDArray[np.floating], 
                         mean: NDArray[np.floating], 
                         cov_inv: NDArray[np.floating]) -> float:
    """
    Compute Mahalanobis distance: √((x-μ)ᵀ Σ⁻¹ (x-μ))
    
    Parameters
    ----------
    x : NDArray
        Data point
    mean : NDArray
        Mean vector
    cov_inv : NDArray
        Inverse covariance matrix
        
    Returns
    -------
    float
        Mahalanobis distance
    """
    diff = x - mean
    return float(diff @ cov_inv @ diff)


# ========================
# Prefix Sum Utilities
# ========================

def build_prefix_sum_1d(x: NDArray) -> NDArray:
    """
    Build prefix sum array for O(1) range sum queries.
    
    Parameters
    ----------
    x : NDArray
        1D input array
        
    Returns
    -------
    NDArray
        Prefix sum array of length len(x)+1 where prefix[i] = sum(x[:i])
    """
    prefix = np.zeros(x.size + 1, dtype=x.dtype)
    np.cumsum(x, out=prefix[1:])
    return prefix


def build_prefix_sum_2d(X: NDArray) -> NDArray:
    """
    Build 2D prefix sum array for O(1) rectangular range sum queries.
    
    Parameters
    ----------
    X : NDArray
        2D input array
        
    Returns
    -------
    NDArray
        2D prefix sum array where prefix[i,j] = sum(X[:i,:j])
    """
    m, n = X.shape
    prefix = np.zeros((m + 1, n + 1), dtype=X.dtype)
    prefix[1:, 1:] = X.cumsum(axis=0).cumsum(axis=1)
    return prefix


def range_sum_1d(prefix: NDArray, start: int, end: int) -> float:
    """
    Query sum of elements in range [start, end) using prefix sum.
    
    Parameters
    ----------
    prefix : NDArray
        Prefix sum array
    start : int
        Start index (inclusive)
    end : int
        End index (exclusive)
        
    Returns
    -------
    float
        Sum of elements in the range
    """
    if start < 0 or end > len(prefix) - 1 or start > end:
        raise ValueError("Invalid range")
    return float(prefix[end] - prefix[start])


def range_sum_2d(prefix: NDArray, r1: int, c1: int, r2: int, c2: int) -> float:
    """
    Query sum of elements in rectangle [(r1,c1), (r2,c2)) using 2D prefix sum.
    
    Parameters
    ----------
    prefix : NDArray
        2D prefix sum array
    r1, c1 : int
        Top-left corner (inclusive)
    r2, c2 : int
        Bottom-right corner (exclusive)
        
    Returns
    -------
    float
        Sum of elements in the rectangle
    """
    if (r1 < 0 or c1 < 0 or r2 > prefix.shape[0] - 1 or 
        c2 > prefix.shape[1] - 1 or r1 > r2 or c1 > c2):
        raise ValueError("Invalid range")
    return float(prefix[r2, c2] - prefix[r2, c1] - prefix[r1, c2] + prefix[r1, c1])


# ========================
# Time Series Utilities
# ========================

def sliding_window(X: NDArray, window_size: int, step: int = 1) -> NDArray:
    """
    Create sliding windows from a time series.
    
    Parameters
    ----------
    X : NDArray
        Input time series
    window_size : int
        Size of each window
    step : int
        Step size between windows
        
    Returns
    -------
    NDArray
        Array of windows, shape (n_windows, window_size, ...)
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    n_windows = max(0, (n_samples - window_size) // step + 1)
    
    if n_windows == 0:
        return np.array([])
    
    windows = np.empty((n_windows, window_size, n_features), dtype=X.dtype)
    
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        windows[i] = X[start:end]
    
    return windows


def circular_shift(x: NDArray, shift: int) -> NDArray:
    """
    Circular shift array by 'shift' positions.
    
    Parameters
    ----------
    x : NDArray
        Input array
    shift : int
        Number of positions to shift (positive = left shift)
        
    Returns
    -------
    NDArray
        Shifted array
    """
    shift = shift % len(x)
    return np.concatenate((x[shift:], x[:shift]))


T = TypeVar('T')
def circular_distance(a: int, b: int, N: int) -> int:
    """
    Compute shortest circular distance between two positions on a circle of size N.
    
    Parameters
    ----------
    a, b : int
        Positions on the circle (0 to N-1)
    N : int
        Circle size
        
    Returns
    -------
    int
        Shortest circular distance
    """
    return min((b - a) % N, (a - b) % N)


def autocorr(x: NDArray[np.floating], max_lag: Optional[int] = None) -> NDArray[np.floating]:
    """
    Compute autocorrelation of a time series up to max_lag.
    
    Parameters
    ----------
    x : NDArray
        Input time series
    max_lag : Optional[int]
        Maximum lag to compute (default: N//2)
        
    Returns
    -------
    NDArray
        Autocorrelation values for lags 0 to max_lag
    """
    n = len(x)
    if max_lag is None:
        max_lag = n // 2
    
    # Center the data
    x_centered = x - np.mean(x)
    
    # Compute variance (denominator)
    var = np.var(x_centered)
    if var == 0:
        return np.ones(max_lag + 1)
    
    # Compute autocorrelation
    acorr = np.zeros(max_lag + 1)
    acorr[0] = 1.0  # Lag 0 is always 1
    
    for lag in range(1, max_lag + 1):
        acorr[lag] = np.sum(x_centered[:-lag] * x_centered[lag:]) / ((n - lag) * var)
    
    return acorr


# ========================
# Sampling Utilities
# ========================

def kmeanspp_init(X: NDArray[np.floating], 
                 k: int, 
                 random_state: Optional[int] = None) -> NDArray[np.floating]:
    """
    Initialize cluster centers using k-means++ algorithm.
    
    Parameters
    ----------
    X : NDArray
        Data points of shape (n_samples, n_features)
    k : int
        Number of clusters
    random_state : Optional[int]
        Random seed
        
    Returns
    -------
    NDArray
        Initial cluster centers of shape (k, n_features)
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples, n_features = X.shape
    centers = np.empty((k, n_features), dtype=X.dtype)
    
    # Choose first center randomly
    centers[0] = X[np.random.randint(n_samples)]
    
    # Choose remaining centers
    for i in range(1, k):
        # Compute squared distances to nearest existing center
        min_dists = np.min([np.sum((X - centers[j])**2, axis=1) 
                            for j in range(i)], axis=0)
        
        # Choose next center with probability proportional to squared distance
        probs = min_dists / np.sum(min_dists)
        centers[i] = X[np.random.choice(n_samples, p=probs)]
    
    return centers


def random_orthogonal_matrix(n: int, random_state: Optional[int] = None) -> NDArray[np.floating]:
    """
    Generate random orthogonal matrix via QR decomposition.
    
    Parameters
    ----------
    n : int
        Matrix size
    random_state : Optional[int]
        Random seed
        
    Returns
    -------
    NDArray
        Random orthogonal matrix of shape (n, n)
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    A = np.random.randn(n, n)
    Q, R = np.linalg.qr(A)
    
    # Ensure determinant is 1 (proper rotation)
    D = np.diag(np.sign(np.diag(R)))
    return Q @ D


# ========================
# Median Heuristic
# ========================

def median_heuristic(X: NDArray[np.floating], 
                    subsample: int = 1000, 
                    random_state: Optional[int] = None) -> float:
    """
    Compute median heuristic for RBF kernel bandwidth.
    
    Parameters
    ----------
    X : NDArray
        Data points of shape (n_samples, n_features)
    subsample : int
        Maximum number of points to use (for efficiency)
    random_state : Optional[int]
        Random seed
        
    Returns
    -------
    float
        Suggested bandwidth parameter
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = X.shape[0]
    
    if n_samples <= subsample:
        # Use all data points
        idx = np.arange(n_samples)
    else:
        # Subsample for efficiency
        idx = np.random.choice(n_samples, size=subsample, replace=False)
    
    X_sub = X[idx]
    
    # Compute pairwise squared distances
    dists = np.sum((X_sub[:, None, :] - X_sub[None, :, :]) ** 2, axis=2)
    
    # Extract upper triangular part (excluding diagonal)
    mask = np.triu(np.ones_like(dists, dtype=bool), k=1)
    dists = dists[mask]
    
    if len(dists) == 0:
        return 1.0
    
    # Return median of non-zero distances
    dists_sqrt = np.sqrt(dists)
    return float(np.median(dists_sqrt))
