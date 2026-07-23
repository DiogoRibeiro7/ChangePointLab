import itertools

import numpy as np
import pytest

from changepoint_lab import KernelCPD
from changepoint_lab.algorithms.kernel import kcp_core as kcp
from changepoint_lab.algorithms.kernel.kcp_rff import (
    RFFConfig,
    build_feature_prefix,
    rbf_rff_map,
    rff_kcp_penalized,
)


def _kernel_cost(K: np.ndarray, start: int, stop: int) -> float:
    block = K[start:stop, start:stop]
    return float(np.trace(block) - np.sum(block) / (stop - start))


def _bruteforce_fixed_m(K: np.ndarray, *, m: int, min_size: int) -> tuple[list[int], float]:
    n = K.shape[0]
    best_cps: list[int] | None = None
    best_cost = float("inf")
    for cps in itertools.combinations(range(1, n), m - 1):
        edges = [0, *cps, n]
        if any(b - a < min_size for a, b in zip(edges[:-1], edges[1:], strict=True)):
            continue
        cost = sum(_kernel_cost(K, a, b) for a, b in zip(edges[:-1], edges[1:], strict=True))
        if cost < best_cost:
            best_cost = cost
            best_cps = list(cps)
    assert best_cps is not None
    return best_cps, best_cost


def test_kcp_respects_max_seg_len():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 1))
    K = kcp.gram_linear(X)
    pref = kcp.build_kernel_prefix(K)
    res = kcp.kcp_penalized(pref, penalty=0.0, min_size=1, method="op", max_seg_len=2)
    lengths = np.diff(res.edges)
    assert np.all(lengths <= 2)


def test_kcp_handles_large_grid_jump():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 1))
    K = kcp.gram_linear(X)
    pref = kcp.build_kernel_prefix(K)
    res = kcp.kcp_penalized(pref, penalty=1.0, min_size=3, grid_jump=50)
    assert res.change_points.size == 0


def test_kcp_backtracking_returns_interior_right_exclusive_boundaries():
    X = np.array([[0.0], [0.0], [5.0], [5.0]])
    K = kcp.gram_linear(X)
    pref = kcp.build_kernel_prefix(K)
    expected_cps, expected_cost = _bruteforce_fixed_m(K, m=2, min_size=1)

    fixed = kcp.kcp_fixed_m(pref, m=2, min_size=1)
    penalized = kcp.kcp_penalized(pref, penalty=0.1, min_size=1, method="op")
    wrapper = KernelCPD(penalty=0.1, kernel=kcp.gram_linear, min_size=1, method="op").fit_predict(X)

    assert expected_cps == [2]
    assert fixed.change_points.tolist() == expected_cps
    assert fixed.edges.tolist() == [0, 2, 4]
    assert fixed.labels.tolist() == [0, 0, 1, 1]
    assert fixed.total_cost == pytest.approx(expected_cost)
    assert penalized.change_points.tolist() == expected_cps
    assert penalized.edges.tolist() == [0, 2, 4]
    assert wrapper.indices.tolist() == expected_cps
    assert wrapper.metadata["edges"].tolist() == [0, 2, 4]


def test_rff_kcp_backtracking_drops_terminal_endpoint():
    X = np.array([[0.0], [0.0], [5.0], [5.0]])
    rff = rbf_rff_map(X, RFFConfig(n_features=4, seed=0), gamma=0.5)
    res = rff_kcp_penalized(
        build_feature_prefix(rff.Z),
        gamma_pen=0.1,
        min_size=1,
        method="op",
    )

    assert res.change_points.tolist() == [2]
    assert res.edges.tolist() == [0, 2, 4]
    assert res.labels.tolist() == [0, 0, 1, 1]
