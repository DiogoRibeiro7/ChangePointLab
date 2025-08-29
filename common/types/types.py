# types.py
# MIT License
"""
Common type definitions for the Change-Point & State-Space Toolkit.
Centralizes shared types to avoid circular imports.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Common type aliases
Array1D = NDArray[np.int64]
Array1DFloat = NDArray[np.floating]
ArrayBool = NDArray[np.bool_]
Tau = Tuple[int, ...]  # Sorted tuple of changepoint positions

# Common dataclasses for configurations and results
@dataclass(frozen=True)
class RJConfig:
    """
    Reversible-jump MCMC configuration.

    Attributes
    ----------
    iters : int
        Total number of MCMC iterations.
    burn : int
        Burn-in iterations (discarded).
    thin : int
        Keep one sample every `thin` iterations after burn-in (>=1).
    seed : Optional[int]
        PRNG seed for reproducibility.
    move_prob : float
        Probability for a 'move' proposal when m>1.
    birth_prob : float
        Probability for a 'birth' proposal (adding a changepoint).
    death_prob : float
        Probability for a 'death' proposal (deleting a changepoint).
        (move_prob + birth_prob + death_prob) must equal 1 for m>1.
        For m==1, only birth proposals are used.
    """

    iters: int = 30_000
    burn: int = 10_000
    thin: int = 10
    seed: Optional[int] = 7
    move_prob: float = 0.50
    birth_prob: float = 0.25
    death_prob: float = 0.25

    def __post_init__(self) -> None:
        if self.iters <= 0 or self.burn < 0 or self.thin <= 0:
            raise ValueError("iters>0, burn>=0, thin>0 required.")
        if self.burn >= self.iters:
            raise ValueError("burn must be < iters.")


@dataclass(frozen=True)
class PTConfig:
    """
    Parallel Tempering configuration for two chains (cold T=1, hot T>1).
    
    Attributes
    ----------
    iters : int
        Total number of MCMC iterations.
    burn : int
        Burn-in iterations (discarded).
    thin : int
        Keep one sample every `thin` iterations after burn-in (>=1).
    swap_every : int
        Number of iterations between swap attempts.
    T_hot : float
        Temperature of the hot chain (T>1).
    seed : Optional[int]
        PRNG seed for reproducibility.
    """
    iters: int = 20_000
    burn: int = 10_000
    thin: int = 10
    swap_every: int = 50
    T_hot: float = 3.0
    seed: Optional[int] = 123

    def __post_init__(self) -> None:
        if self.iters <= 0 or self.burn < 0 or self.thin <= 0:
            raise ValueError("iters>0, burn>=0, thin>0 required.")
        if self.burn >= self.iters:
            raise ValueError("burn must be < iters.")
        if self.swap_every <= 0:
            raise ValueError("swap_every must be positive.")
        if self.T_hot <= 1.0:
            raise ValueError("T_hot must be greater than 1.0.")


@dataclass
class MCMCResult:
    """
    Container for MCMC outputs.

    Attributes
    ----------
    samples_tau : List[Tau]
        Posterior samples of changepoint vectors (post-burn, thinned).
    log_posteriors : List[float]
        Log unnormalized posterior values for each kept sample.
    changepoint_hist : Array1D
        Histogram counts for positions 0..N-1 across samples.
    mode_tau : Tau
        The MAP (highest log posterior within kept samples).
    """

    samples_tau: List[Tau]
    log_posteriors: List[float]
    changepoint_hist: Array1D
    mode_tau: Tau


@dataclass
class PTResult:
    """
    Container for Parallel Tempering outputs.
    
    Attributes
    ----------
    samples_tau_cold : List[Tau]
        Posterior samples from the cold chain.
    log_posts_cold : List[float]
        Log posterior values for cold chain samples.
    cp_hist_cold : Array1D
        Changepoint histogram from cold chain samples.
    mode_tau_cold : Tau
        MAP estimate from cold chain.
    swaps_attempted : int
        Number of swap attempts.
    swaps_accepted : int
        Number of accepted swaps.
    """
    samples_tau_cold: List[Tau]
    log_posts_cold: List[float]
    cp_hist_cold: Array1D
    mode_tau_cold: Tau
    swaps_attempted: int
    swaps_accepted: int


# Common result interfaces for standardized returns across algorithms
@dataclass
class ChangePointResult:
    """
    Standardized result container for change-point detection algorithms.
    
    Attributes
    ----------
    change_points : List[int]
        Detected change point positions.
    segments : List[Tuple[int, int]]
        Segments as (start, end) pairs.
    scores : Optional[List[float]]
        Optional scores/statistics for each change point.
    cost : float
        Objective function value.
    model_name : str
        Name of the model/algorithm used.
    parameters : Dict
        Algorithm parameters used.
    """
    change_points: List[int]
    segments: List[Tuple[int, int]]
    scores: Optional[List[float]] = None
    cost: float = 0.0
    model_name: str = ""
    parameters: Dict = None
    
    def __post_init__(self):
        if self.parameters is None:
            object.__setattr__(self, "parameters", {})
