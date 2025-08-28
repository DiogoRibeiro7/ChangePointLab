# bb_utils.py

from __future__ import annotations
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from bayesian_blocks import BBResult

def blocks_to_labels_index(N: int, result: BBResult) -> NDArray[np.floating]:
    """
    Expand Bayesian Blocks (index space) into a length-N stepwise array.

    Parameters
    ----------
    N : int
        Number of cells/samples.
    result : BBResult
        Output from bayesian_blocks_counts(...) or bayesian_blocks_bernoulli(...).

    Returns
    -------
    yhat : np.ndarray, shape (N,)
        Piecewise-constant block values per index.
    """
    edges = result.edges.astype(int)
    vals = result.block_value
    if edges.size != vals.size + 1:
        raise ValueError("edges and block_value sizes are inconsistent.")
    yhat = np.empty(N, dtype=float)
    for k in range(vals.size):
        a, b = edges[k], edges[k + 1]
        yhat[a:b] = vals[k]
    return yhat


def blocks_to_labels_time(t: NDArray[np.floating], result: BBResult) -> NDArray[np.floating]:
    """
    Expand Bayesian Blocks (time space) onto sample timestamps t (monotone).

    Each t[i] is assigned the block value whose [edge_k, edge_{k+1}) contains t[i].

    Parameters
    ----------
    t : array of shape (N,)
        Monotone non-decreasing sample times.
    result : BBResult
        Output from bayesian_blocks_events(...).

    Returns
    -------
    yhat : np.ndarray, shape (N,)
        Stepwise block values aligned to t.
    """
    edges = result.edges
    vals = result.block_value
    if edges.size != vals.size + 1:
        raise ValueError("edges and block_value sizes are inconsistent.")
    idx = np.searchsorted(edges[1:], t, side="right")  # block index for each t
    return vals[idx]
