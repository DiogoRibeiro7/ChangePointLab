# api_harmonizer.py
# MIT License
"""
API Harmonizer for the Change-Point & State-Space Toolkit.

This utility provides adapter functions to standardize the interfaces
across different modules, ensuring consistent return types and parameter
naming conventions. It enables seamless integration of various algorithms
through a unified API.
"""

from __future__ import annotations
import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Import shared types
from common.types.types import ChangePointResult, Tau, Array1D, Array1DFloat, ArrayBool

# Import algorithm modules
try:
    import bayesian_blocks
    import edivisive
    import kcp
    import kcp_rff
    import within_period.within_period_cpd as within_period_cpd
    import hsmm
    import sdhmm

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


class AlgorithmRegistry:
    """
    Registry of standardized algorithm interfaces.

    This class maintains a registry of all available algorithms with
    standardized parameter names and return types for consistent access.
    """

    def __init__(self):
        self.algorithms = {}
        self.categories = {}
        if MODULES_AVAILABLE:
            self._register_algorithms()

    def _register_algorithms(self):
        """Register all available algorithms with standardized interfaces."""
        # Register change-point detection algorithms
        self._register_bayesian_blocks()
        self._register_edivisive()
        self._register_kcp()
        self._register_kcp_rff()
        self._register_within_period()

        # Register state-space models
        self._register_hsmm()
        self._register_sdhmm()

    def _register_bayesian_blocks(self):
        """Register Bayesian Blocks algorithms."""
        self.categories["bayesian_blocks"] = "change_point"

        # Events mode
        self.algorithms["bayesian_blocks_events"] = {
            "function": self.bayesian_blocks_events_adapter,
            "description": "Bayesian Blocks for event data (unbinned Poisson)",
            "params": {
                "data": "Event times",
                "t_start": "Start time (optional)",
                "t_stop": "End time (optional)",
                "p0": "False positive rate prior",
            },
            "returns": "ChangePointResult with block edges and values",
        }

        # Counts mode
        self.algorithms["bayesian_blocks_counts"] = {
            "function": self.bayesian_blocks_counts_adapter,
            "description": "Bayesian Blocks for binned count data",
            "params": {
                "data": "Count data array",
                "widths": "Bin widths (optional)",
                "p0": "False positive rate prior",
            },
            "returns": "ChangePointResult with block edges and values",
        }

        # Bernoulli mode
        self.algorithms["bayesian_blocks_bernoulli"] = {
            "function": self.bayesian_blocks_bernoulli_adapter,
            "description": "Bayesian Blocks for Bernoulli/binary data",
            "params": {
                "data": "Binary success/fail data",
                "trials": "Number of trials (optional)",
                "p0": "False positive rate prior",
            },
            "returns": "ChangePointResult with block edges and values",
        }

    def _register_edivisive(self):
        """Register E-Divisive algorithm."""
        self.categories["edivisive"] = "change_point"

        self.algorithms["edivisive"] = {
            "function": self.edivisive_adapter,
            "description": "E-Divisive multivariate change-point detection",
            "params": {
                "data": "Multivariate time series",
                "alpha": "Energy statistic parameter",
                "min_size": "Minimum segment size",
                "R": "Number of permutations",
                "significance": "Significance level",
                "resample": "Resampling method",
                "block_size": "Block size for bootstrap",
                "seed": "Random seed",
            },
            "returns": "ChangePointResult with detected change points",
        }

    def _register_kcp(self):
        """Register Kernel Change-Point detection algorithms."""
        self.categories["kcp"] = "change_point"

        # Penalized KCP
        self.algorithms["kcp_penalized"] = {
            "function": self.kcp_penalized_adapter,
            "description": "Penalized Kernel Change-Point Detection",
            "params": {
                "data": "Time series data",
                "kernel": "Kernel type ('rbf', 'linear')",
                "gamma": "Kernel parameter/penalty",
                "min_size": "Minimum segment size",
                "method": "Optimization method ('pelt', 'op')",
            },
            "returns": "ChangePointResult with detected change points",
        }

        # Fixed-m KCP
        self.algorithms["kcp_fixed_m"] = {
            "function": self.kcp_fixed_m_adapter,
            "description": "Kernel Change-Point Detection with fixed number of segments",
            "params": {
                "data": "Time series data",
                "kernel": "Kernel type ('rbf', 'linear')",
                "m": "Number of segments",
                "min_size": "Minimum segment size",
            },
            "returns": "ChangePointResult with detected change points",
        }

    def _register_kcp_rff(self):
        """Register RFF-based Kernel Change-Point detection algorithms."""
        self.categories["kcp_rff"] = "change_point"

        # Penalized RFF KCP
        self.algorithms["rff_kcp_penalized"] = {
            "function": self.rff_kcp_penalized_adapter,
            "description": "Penalized Random Fourier Features Kernel Change-Point Detection",
            "params": {
                "data": "Time series data",
                "n_features": "Number of random features",
                "gamma": "Kernel parameter/penalty",
                "min_size": "Minimum segment size",
                "method": "Optimization method ('pelt', 'op')",
                "rff_type": "RFF variant ('standard', 'orthogonal', 'quasi_mc')",
            },
            "returns": "ChangePointResult with detected change points",
        }

        # Fixed-m RFF KCP
        self.algorithms["rff_kcp_fixed_m"] = {
            "function": self.rff_kcp_fixed_m_adapter,
            "description": "RFF Kernel Change-Point Detection with fixed number of segments",
            "params": {
                "data": "Time series data",
                "n_features": "Number of random features",
                "m": "Number of segments",
                "min_size": "Minimum segment size",
                "rff_type": "RFF variant ('standard', 'orthogonal', 'quasi_mc')",
            },
            "returns": "ChangePointResult with detected change points",
        }

    def _register_within_period(self):
        """Register Within-Period Change-Point Detection."""
        self.categories["within_period_cpd"] = "change_point"

        self.algorithms["within_period_cpd"] = {
            "function": self.within_period_adapter,
            "description": "Within-Period Change-Point Detection for periodic binary data",
            "params": {
                "data": "Binary time series",
                "N": "Period length",
                "l": "Minimum segment length",
                "iters": "MCMC iterations",
                "burn": "Burn-in iterations",
                "thin": "Thinning interval",
                "seed": "Random seed",
                "tempering": "Use parallel tempering",
            },
            "returns": "ChangePointResult with detected change points and posterior samples",
        }

    def _register_hsmm(self):
        """Register Hidden Semi-Markov Model."""
        self.categories["hsmm"] = "state_space"

        self.algorithms["hsmm"] = {
            "function": self.hsmm_adapter,
            "description": "Hidden Semi-Markov Model with explicit durations",
            "params": {
                "data": "Time series data",
                "n_states": "Number of hidden states",
                "emission_type": "Emission distribution type",
                "max_duration": "Maximum state duration",
                "min_duration": "Minimum state duration",
                "duration_type": "Duration distribution type",
                "max_iter": "Maximum EM iterations",
            },
            "returns": "Dictionary with states, durations, and model parameters",
        }

    def _register_sdhmm(self):
        """Register Scaled-Dirichlet Hidden Markov Model."""
        self.categories["sdhmm"] = "state_space"

        self.algorithms["sdhmm"] = {
            "function": self.sdhmm_adapter,
            "description": "Scaled-Dirichlet Hidden Markov Model",
            "params": {
                "data": "Time series data",
                "n_states": "Number of hidden states",
                "n_components": "Number of mixture components per state",
                "max_iter": "Maximum iterations",
            },
            "returns": "Dictionary with states and model parameters",
        }

    # ========== Adapter Functions ==========

    def bayesian_blocks_events_adapter(
        self,
        data: NDArray,
        t_start: Optional[float] = None,
        t_stop: Optional[float] = None,
        p0: float = 0.05,
    ) -> ChangePointResult:
        """
        Adapter for Bayesian Blocks events method.

        Parameters
        ----------
        data : NDArray
            Event times
        t_start : Optional[float]
            Start time
        t_stop : Optional[float]
            End time
        p0 : float
            False positive rate prior

        Returns
        -------
        ChangePointResult
            Standardized result
        """
        if not MODULES_AVAILABLE:
            raise ImportError("bayesian_blocks module not available")

        # Call original function
        result = bayesian_blocks.bayesian_blocks_events(data, t_start, t_stop, p0)

        # Convert to standard result
        change_points = list(result.edges[1:-1])  # Exclude boundaries
        segments = []
        for i in range(len(result.edges) - 1):
            segments.append((result.edges[i], result.edges[i + 1]))

        return ChangePointResult(
            change_points=change_points,
            segments=segments,
            scores=None,
            cost=0.0,
            model_name="bayesian_blocks_events",
            parameters={"p0": p0},
        )

    def bayesian_blocks_counts_adapter(
        self, data: NDArray, widths: Optional[NDArray] = None, p0: float = 0.05
    ) -> ChangePointResult:
        """
        Adapter for Bayesian Blocks counts method.

        Parameters
        ----------
        data : NDArray
            Count data
        widths : Optional[NDArray]
            Bin widths
        p0 : float
            False positive rate prior

        Returns
        -------
        ChangePointResult
            Standardized result
        """
        if not MODULES_AVAILABLE:
            raise ImportError("bayesian_blocks module not available")

        # Call original function
        result = bayesian_blocks.bayesian_blocks_counts(data, widths, p0)

        # Convert to standard result
        change_points = list(result.edges[1:-1])  # Exclude boundaries
        segments = []
        for i in range(len(result.edges) - 1):
            segments.append((result.edges[i], result.edges[i + 1]))

        return ChangePointResult(
            change_points=change_points,
            segments=segments,
            scores=None,
            cost=0.0,
            model_name="bayesian_blocks_counts",
            parameters={"p0": p0},
        )

    def bayesian_blocks_bernoulli_adapter(
        self, data: NDArray, trials: Optional[NDArray] = None, p0: float = 0.05
    ) -> ChangePointResult:
        """
        Adapter for Bayesian Blocks Bernoulli method.

        Parameters
        ----------
        data : NDArray
            Binary success/fail data
        trials : Optional[NDArray]
            Number of trials
        p0 : float
            False positive rate prior

        Returns
        -------
        ChangePointResult
            Standardized result
        """
        if not MODULES_AVAILABLE:
            raise ImportError("bayesian_blocks module not available")

        # Call original function
        result = bayesian_blocks.bayesian_blocks_bernoulli(data, trials, p0)

        # Convert to standard result
        change_points = list(result.edges[1:-1])  # Exclude boundaries
        segments = []
        for i in range(len(result.edges) - 1):
            segments.append((result.edges[i], result.edges[i + 1]))

        return ChangePointResult(
            change_points=change_points,
            segments=segments,
            scores=None,
            cost=0.0,
            model_name="bayesian_blocks_bernoulli",
            parameters={"p0": p0},
        )

    def edivisive_adapter(
        self,
        data: NDArray,
        alpha: float = 1.0,
        min_size: int = 30,
        R: int = 499,
        significance: float = 0.05,
        resample: str = "circular-block-bootstrap",
        block_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> ChangePointResult:
        """
        Adapter for E-Divisive algorithm.

        Parameters
        ----------
        data : NDArray
            Multivariate time series
        alpha : float
            Energy statistic parameter
        min_size : int
            Minimum segment size
        R : int
            Number of permutations
        significance : float
            Significance level
        resample : str
            Resampling method
        block_size : Optional[int]
            Block size for bootstrap
        seed : Optional[int]
            Random seed

        Returns
        -------
        ChangePointResult
            Standardized result
        """
        if not MODULES_AVAILABLE:
            raise ImportError("edivisive module not available")

        # Call original function
        result = edivisive.edivisive(
            data, alpha, min_size, R, significance, resample, block_size, seed
        )

        # Convert to standard result
        change_points = list(result.change_points)
        segments = []
        cp_with_bounds = [0] + change_points + [data.shape[0]]
        for i in range(len(cp_with_bounds) - 1):
            segments.append((cp_with_bounds[i], cp_with_bounds[i + 1]))

        return ChangePointResult(
            change_points=change_points,
            segments=segments,
            scores=list(result.test_statistics) if result.test_statistics is not None else None,
            cost=0.0,
            model_name="edivisive",
            parameters={
                "alpha": alpha,
                "min_size": min_size,
                "R": R,
                "significance": significance,
                "resample": resample,
            },
        )

    def kcp_penalized_adapter(
        self,
        data: NDArray,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        min_size: int = 20,
        method: str = "pelt",
    ) -> ChangePointResult:
        """
        Adapter for penalized Kernel Change-Point Detection.

        Parameters
        ----------
        data : NDArray
            Time series data
        kernel : str
            Kernel type ('rbf', 'linear')
        gamma : Optional[float]
            Kernel parameter/penalty
        min_size : int
            Minimum segment size
        method : str
            Optimization method ('pelt', 'op')

        Returns
        -------
        ChangePointResult
            Standardized result
        """
        if not MODULES_AVAILABLE:
            raise ImportError("kcp module not available")

        # Build kernel matrix
        if kernel == "rbf":
            K, gamma_used = kcp.gram_rbf(data, gamma=gamma)
        elif kernel == "linear":
            K = kcp.gram_linear(data)
            gamma_used = gamma
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        # Build prefix sums
        prefix = kcp.build_kernel_prefix(K)

        # Call original function
        result = kcp.kcp_penalized(prefix, gamma=gamma_used, min_size=min_size, method=method)

        # Convert to standard result
        change_points = list(result.change_points)
        segments = []
        cp_with_bounds = [0] + change_points + [data.shape[0]]
        for i in range(len(cp_with_bounds) - 1):
            segments.append((cp_with_bounds[i], cp_with_bounds[i + 1]))

        return ChangePointResult(
            change_points=change_points,
            segments=segments,
            scores=None,
            cost=float(result.cost),
            model_name="kcp_penalized",
            parameters={
                "kernel": kernel,
                "gamma": gamma_used,
                "min_size": min_size,
                "method": method,
            },
        )

    def kcp_fixed_m_adapter(
        self, data: NDArray, kernel: str = "rbf", m: int = 2, min_size: int = 20
    ) -> ChangePointResult:
        """
        Adapter for Kernel Change-Point Detection with fixed number of segments.

        Parameters
        ----------
        data : NDArray
            Time series data
        kernel : str
            Kernel type ('rbf', 'linear')
        m : int
            Number of segments
        min_size : int
            Minimum segment size

        Returns
        -------
        ChangePointResult
            Standardized result
        """
        if not MODULES_AVAILABLE:
            raise ImportError("kcp module not available")

        # Build kernel matrix
        if kernel == "rbf":
            K, gamma_used = kcp.gram_rbf(data)
        elif kernel == "linear":
            K = kcp.gram_linear(data)
            gamma_used = None
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        # Build prefix sums
        prefix = kcp.build_kernel_prefix(K)

        # Call original function
        result = kcp.kcp_fixed_m(prefix, m=m, min_size=min_size)

        # Convert to standard result
        change_points = list(result.change_points)
        segments = []
        cp_with_bounds = [0] + change_points + [data.shape[0]]
        for i in range(len(cp_with_bounds) - 1):
            segments.append((cp_with_bounds[i], cp_with_bounds[i + 1]))

        return ChangePointResult(
            change_points=change_points,
            segments=segments,
            scores=None,
            cost=float(result.cost),
            model_name="kcp_fixed_m",
            parameters={"kernel": kernel, "m": m, "min_size": min_size},
        )

    def rff_kcp_penalized_adapter(
        self,
        data: NDArray,
        n_features: int = 512,
        gamma: Optional[float] = None,
        min_size: int = 20,
        method: str = "pelt",
        rff_type: str = "standard",
    ) -> ChangePointResult:
        """
        Adapter for penalized RFF Kernel Change-Point Detection.

        Parameters
        ----------
        data : NDArray
            Time series data
        n_features : int
            Number of random features
        gamma : Optional[float]
            Kernel parameter/penalty
        min_size : int
            Minimum segment size
        method : str
            Optimization method ('pelt', 'op')
        rff_type : str
            RFF variant ('standard', 'orthogonal', 'quasi_mc')

        Returns
        -------
        ChangePointResult
            Standardized result
        """
        if not MODULES_AVAILABLE:
            raise ImportError("kcp_rff module not available")

        # Import the appropriate RFF variant
        if rff_type == "standard":
            from kcp_rff import RFFConfig, rbf_rff_map

            rff_config = RFFConfig(n_features=n_features, seed=42)
        elif rff_type == "orthogonal":
            from kcp.rff_variants import OrthogonalRFFConfig, orthogonal_rff_map as rbf_rff_map

            rff_config = OrthogonalRFFConfig(n_features=n_features, seed=42)
        elif rff_type == "quasi_mc":
            from kcp.rff_variants import QuasiMCRFFConfig, quasi_mc_rff_map as rbf_rff_map

            rff_config = QuasiMCRFFConfig(n_features=n_features, seed=42)
        else:
            raise ValueError(f"Unknown RFF type: {rff_type}")

        # Build RFF mapping
        rff = rbf_rff_map(data, rff_config, gamma=gamma)

        # Build prefix sums
        from kcp_rff import build_feature_prefix, rff_kcp_penalized

        prefix = build_feature_prefix(rff.Z)

        # Call original function
        result = rff_kcp_penalized(prefix, gamma_pen=gamma, min_size=min_size, method=method)

        # Convert to standard result
        change_points = list(result.change_points)
        segments = []
        cp_with_bounds = [0] + change_points + [data.shape[0]]
        for i in range(len(cp_with_bounds) - 1):
            segments.append((cp_with_bounds[i], cp_with_bounds[i + 1]))

        return ChangePointResult(
            change_points=change_points,
            segments=segments,
            scores=None,
            cost=float(result.cost),
            model_name="rff_kcp_penalized",
            parameters={
                "n_features": n_features,
                "gamma": rff.gamma,
                "min_size": min_size,
                "method": method,
                "rff_type": rff_type,
            },
        )

    def rff_kcp_fixed_m_adapter(
        self,
        data: NDArray,
        n_features: int = 512,
        m: int = 2,
        min_size: int = 20,
        rff_type: str = "standard",
    ) -> ChangePointResult:
        """
        Adapter for RFF Kernel Change-Point Detection with fixed number of segments.

        Parameters
        ----------
        data : NDArray
            Time series data
        n_features : int
            Number of random features
        m : int
            Number of segments
        min_size : int
            Minimum segment size
        rff_type : str
            RFF variant ('standard', 'orthogonal', 'quasi_mc')

        Returns
        -------
        ChangePointResult
            Standardized result
        """
        if not MODULES_AVAILABLE:
            raise ImportError("kcp_rff module not available")

        # Import the appropriate RFF variant
        if rff_type == "standard":
            from kcp_rff import RFFConfig, rbf_rff_map

            rff_config = RFFConfig(n_features=n_features, seed=42)
        elif rff_type == "orthogonal":
            from kcp.rff_variants import OrthogonalRFFConfig, orthogonal_rff_map as rbf_rff_map

            rff_config = OrthogonalRFFConfig(n_features=n_features, seed=42)
        elif rff_type == "quasi_mc":
            from kcp.rff_variants import QuasiMCRFFConfig, quasi_mc_rff_map as rbf_rff_map

            rff_config = QuasiMCRFFConfig(n_features=n_features, seed=42)
        else:
            raise ValueError(f"Unknown RFF type: {rff_type}")

        # Build RFF mapping
        rff = rbf_rff_map(data, rff_config)

        # Build prefix sums
        from kcp_rff import build_feature_prefix, rff_kcp_fixed_m

        prefix = build_feature_prefix(rff.Z)

        # Call original function
        result = rff_kcp_fixed_m(prefix, m=m, min_size=min_size)

        # Convert to standard result
        change_points = list(result.change_points)
        segments = []
        cp_with_bounds = [0] + change_points + [data.shape[0]]
        for i in range(len(cp_with_bounds) - 1):
            segments.append((cp_with_bounds[i], cp_with_bounds[i + 1]))

        return ChangePointResult(
            change_points=change_points,
            segments=segments,
            scores=None,
            cost=float(result.cost),
            model_name="rff_kcp_fixed_m",
            parameters={
                "n_features": n_features,
                "gamma": rff.gamma,
                "min_size": min_size,
                "m": m,
                "rff_type": rff_type,
            },
        )

    def within_period_adapter(
        self,
        data: NDArray,
        N: int,
        l: int = 4,
        iters: int = 20000,
        burn: int = 10000,
        thin: int = 10,
        seed: Optional[int] = None,
        tempering: bool = False,
    ) -> ChangePointResult:
        """
        Adapter for Within-Period Change-Point Detection.

        Parameters
        ----------
        data : NDArray
            Binary time series
        N : int
            Period length
        l : int
            Minimum segment length
        iters : int
            MCMC iterations
        burn : int
            Burn-in iterations
        thin : int
            Thinning interval
        seed : Optional[int]
            Random seed
        tempering : bool
            Use parallel tempering

        Returns
        -------
        ChangePointResult
            Standardized result
        """
        if not MODULES_AVAILABLE:
            raise ImportError("within_period_cpd module not available")

        # Set up prior and model
        prior = within_period_cpd.ModelPrior(N=N, l=l, gamma=1.0, pois_lambda=1.0)
        model = within_period_cpd.WithinPeriodCPD(prior)

        if not tempering:
            # Use standard RJMCMC
            cfg = within_period_cpd.RJConfig(iters=iters, burn=burn, thin=thin, seed=seed)
            result = model.fit(data, cfg)

            # Extract results
            change_points = list(result.mode_tau)
            samples = result.samples_tau
            log_posts = result.log_posteriors
            cp_hist = result.changepoint_hist

        else:
            # Use parallel tempering
            from within_period.samplers.tempering import PTConfig, parallel_tempering_fit

            ptcfg = PTConfig(iters=iters, burn=burn, thin=thin, swap_every=50, T_hot=3.0, seed=seed)
            result = parallel_tempering_fit(model, data, ptcfg)

            # Extract results
            change_points = list(result.mode_tau_cold)
            samples = result.samples_tau_cold
            log_posts = result.log_posts_cold
            cp_hist = result.cp_hist_cold

        # Build segments
        segments = []
        if change_points:
            # Sort changepoints (should already be sorted)
            change_points = sorted(change_points)

            # Previous cp, with wraparound
            prev_cp = change_points[-1]
            for cp in change_points:
                # Each segment is (prev_cp, cp] with modulo N wrapping
                length = (cp - prev_cp) % N
                length = N if length == 0 else length
                segments.append((prev_cp, cp))
                prev_cp = cp
        else:
            # Single segment for the whole period
            segments.append((0, N - 1))

        return ChangePointResult(
            change_points=change_points,
            segments=segments,
            scores=None,
            cost=-max(log_posts) if log_posts else 0.0,
            model_name="within_period_cpd",
            parameters={
                "N": N,
                "l": l,
                "tempering": tempering,
                "samples": samples,
                "cp_hist": cp_hist.tolist() if isinstance(cp_hist, np.ndarray) else cp_hist,
            },
        )

    def hsmm_adapter(
        self,
        data: NDArray,
        n_states: int = 3,
        emission_type: str = "gaussian_diag",
        max_duration: int = 100,
        min_duration: int = 1,
        duration_type: str = "poisson",
        max_iter: int = 100,
    ) -> Dict:
        """
        Adapter for Hidden Semi-Markov Model.

        Parameters
        ----------
        data : NDArray
            Time series data
        n_states : int
            Number of hidden states
        emission_type : str
            Emission distribution type
        max_duration : int
            Maximum state duration
        min_duration : int
            Minimum state duration
        duration_type : str
            Duration distribution type
        max_iter : int
            Maximum EM iterations

        Returns
        -------
        Dict
            Dictionary with states, durations, and model parameters
        """
        if not MODULES_AVAILABLE:
            raise ImportError("hsmm module not available")

        # Configure HSMM
        config = hsmm.HSMMConfig(K=n_states, Dmax=max_duration, min_duration=min_duration)

        # Initialize emission model
        if emission_type == "gaussian_diag":
            from gaussian_diag import estimate_by_kmeanspp, gaussian_diag_loglik

            em = estimate_by_kmeanspp(data, n_states, n_init=5, allow_nan=False)
            loglik = gaussian_diag_loglik(data, em)

        elif emission_type == "gaussian_full":
            from hsmm.gaussian_full import GaussianFullEmissions

            em = GaussianFullEmissions(n_states)
            em.initialize_kmeans(data, n_init=5, seed=42)
            loglik = em.compute_loglik(data)

        elif emission_type == "ar":
            from hsmm.ar_emissions import AREmissions

            em = AREmissions(n_states, order=1)
            em.initialize(data, method="kmeans", seed=42)
            loglik = em.compute_loglik(data)

        else:
            raise ValueError(f"Unknown emission type: {emission_type}")

        # Initialize HSMM parameters
        pi0 = np.full(n_states, 1.0 / n_states)
        A0 = np.full((n_states, n_states), 1.0 / (n_states - 1))
        np.fill_diagonal(A0, 0.0)

        # Set up duration distribution
        if duration_type == "poisson":
            from hsmm import PoissonDur

            mean_durations = np.full(n_states, max_duration / 2)
            duration_dist = ("poisson", PoissonDur(lam=mean_durations))
        elif duration_type == "negbin":
            from hsmm import NegBinDur

            mean_durations = np.full(n_states, max_duration / 2)
            # r=5 for moderate overdispersion
            duration_dist = (
                "negbin",
                NegBinDur(r=np.full(n_states, 5), p=5 / (5 + mean_durations)),
            )
        else:
            raise ValueError(f"Unknown duration type: {duration_type}")

        # Initialize HSMM
        model = hsmm.HSMM(config, hsmm.HSMMParams(pi=pi0, A=A0, duration=duration_dist))

        # Fit model
        params_fit, ll_trace = model.fit(loglik, max_iter=max_iter)

        # Decode states
        states, durations = model.decode_viterbi(loglik)

        # Return results
        return {
            "states": states.tolist() if isinstance(states, np.ndarray) else states,
            "durations": durations.tolist() if isinstance(durations, np.ndarray) else durations,
            "log_likelihood": float(ll_trace[-1]) if ll_trace else 0.0,
            "ll_trace": ll_trace,
            "params": {
                "pi": params_fit.pi.tolist()
                if isinstance(params_fit.pi, np.ndarray)
                else params_fit.pi,
                "A": params_fit.A.tolist()
                if isinstance(params_fit.A, np.ndarray)
                else params_fit.A,
                "duration_type": duration_type,
            },
            "model_name": "hsmm",
        }

    def sdhmm_adapter(
        self, data: NDArray, n_states: int = 3, n_components: int = 1, max_iter: int = 100
    ) -> Dict:
        """
        Adapter for Scaled-Dirichlet Hidden Markov Model.

        Parameters
        ----------
        data : NDArray
            Time series data
        n_states : int
            Number of hidden states
        n_components : int
            Number of mixture components per state
        max_iter : int
            Maximum iterations

        Returns
        -------
        Dict
            Dictionary with states and model parameters
        """
        if not MODULES_AVAILABLE:
            raise ImportError("sdhmm module not available")

        if n_components == 1:
            # Single component per state
            from sdhmm import SDHMM, SDHMMConfig

            config = SDHMMConfig(K=n_states, D=data.shape[1], max_iter=max_iter)
            model = SDHMM(config)
            results = model.fit(data)

            # Decode states
            states = model.viterbi(data)

        else:
            # Multiple components per state
            from sdhmm_mix_vi import SDHMMMixVI, SDHMMMixVIConfig

            config = SDHMMMixVIConfig(
                K=n_states, J=n_components, D=data.shape[1], max_iter=max_iter
            )
            model = SDHMMMixVI(config)
            results = model.fit(data)

            # Decode states and components
            states, components = model.viterbi(data)

        # Return results
        return {
            "states": states.tolist() if isinstance(states, np.ndarray) else states,
            "components": (
                components.tolist() if isinstance(components, np.ndarray) else components
            )
            if n_components > 1
            else None,
            "model": model,
            "log_likelihood": float(results["ll"][-1]) if results and "ll" in results else 0.0,
            "model_name": "sdhmm_mix" if n_components > 1 else "sdhmm",
        }

    # ========== Public Interface ==========

    def list_algorithms(self) -> List[str]:
        """List all available algorithms."""
        return sorted(self.algorithms.keys())

    def list_categories(self) -> Dict[str, List[str]]:
        """List algorithms by category."""
        result = {}
        for algo, category in self.categories.items():
            if category not in result:
                result[category] = []
            result[category].append(algo)
        return {k: sorted(v) for k, v in result.items()}

    def get_algorithm_info(self, name: str) -> Dict:
        """Get information about a specific algorithm."""
        if name not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {name}")
        return self.algorithms[name]

    def run(self, name: str, **kwargs) -> Any:
        """
        Run an algorithm with the given parameters.

        Parameters
        ----------
        name : str
            Algorithm name
        **kwargs : dict
            Algorithm parameters

        Returns
        -------
        Any
            Algorithm result
        """
        if name not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {name}")
        return self.algorithms[name]["function"](**kwargs)


# Global registry instance
registry = AlgorithmRegistry()


# ========== Example Usage ==========


def example_usage() -> None:
    """Example of using the harmonized API."""
    import numpy as np
    import matplotlib.pyplot as plt

    # List available algorithms
    print("Available algorithms:")
    for algo in registry.list_algorithms():
        print(f"  - {algo}")

    # List algorithms by category
    print("\nAlgorithms by category:")
    for category, algos in registry.list_categories().items():
        print(f"  {category}:")
        for algo in algos:
            print(f"    - {algo}")

    # Generate synthetic data
    np.random.seed(42)
    N = 500
    x = np.zeros(N)
    x[100:200] = 1.0
    x[300:400] = -1.0
    x = x + np.random.normal(0, 0.3, N)

    # Run KCP with the harmonized API
    result = registry.run("kcp_penalized", data=x, kernel="rbf", gamma=np.log(N), min_size=20)

    print(f"\nDetected change points: {result.change_points}")
    print(f"Number of segments: {len(result.segments)}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(x)
    for cp in result.change_points:
        plt.axvline(cp, color="r", linestyle="--")
    plt.title("KCP Results")
    plt.savefig("kcp_example.png")
    plt.close()

    # Run HSMM with the harmonized API
    X = np.column_stack([x, np.roll(x, 5)])
    hsmm_result = registry.run(
        "hsmm", data=X, n_states=3, emission_type="gaussian_diag", max_duration=50
    )

    print(f"\nHSMM unique states: {np.unique(hsmm_result['states'])}")
    print(f"Final log-likelihood: {hsmm_result['log_likelihood']}")

    # Plot HSMM states
    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.plot(X)
    plt.title("Data")

    plt.subplot(212)
    plt.step(range(len(hsmm_result["states"])), hsmm_result["states"])
    plt.title("HSMM States")
    plt.tight_layout()
    plt.savefig("hsmm_example.png")
    plt.close()


if __name__ == "__main__":
    if MODULES_AVAILABLE:
        example_usage()
    else:
        print("Example requires module imports. Run in the correct environment.")
