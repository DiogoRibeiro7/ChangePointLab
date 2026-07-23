import itertools

import numpy as np
import pytest

from changepoint_lab import KernelCPD
from changepoint_lab.algorithms.kernel.bandwidth_cv import (
    BandwidthCVConfig,
    _compute_changepoint_score,
    _generate_candidate_sigmas,
    _kfold_split,
    _time_series_split,
)
from changepoint_lab.algorithms.kernel.kcp import KernelMatrix
from changepoint_lab.algorithms.kernel.kcp_core import (
    build_kernel_prefix,
    gram_linear,
    gram_rbf,
    kcp_fixed_m,
    kcp_penalized,
    kernel_segment_cost,
)
from changepoint_lab.algorithms.kernel.kcp_rff import RFFConfig, build_feature_prefix, rbf_rff_map


def _feature_cost(X: np.ndarray, start: int, stop: int) -> float:
    segment = X[start:stop]
    centered = segment - segment.mean(axis=0, keepdims=True)
    return float(np.sum(centered * centered))


def _kernel_cost(K: np.ndarray, start: int, stop: int) -> float:
    block = K[start:stop, start:stop]
    return float(np.trace(block) - np.sum(block) / (stop - start))


def _bruteforce_fixed_m(
    n: int,
    *,
    m: int,
    min_size: int,
    cost,
) -> tuple[list[int], float]:
    best_cps: list[int] | None = None
    best_cost = float("inf")
    for cps in itertools.combinations(range(1, n), m - 1):
        edges = [0, *cps, n]
        if any(stop - start < min_size for start, stop in zip(edges[:-1], edges[1:], strict=True)):
            continue
        total = sum(cost(start, stop) for start, stop in zip(edges[:-1], edges[1:], strict=True))
        if total < best_cost:
            best_cps = list(cps)
            best_cost = total
    assert best_cps is not None
    return best_cps, best_cost


def test_kernel_cpd_fit_predict_exact_smoke_and_metadata() -> None:
    X = np.array([[0.0], [0.1], [0.2], [4.0], [4.1], [4.2]])

    result = KernelCPD(penalty=0.1, min_size=2, method="op").fit_predict(X)

    assert result.indices.tolist() == [3]
    assert result.metadata["approximation"] == "exact"
    assert result.metadata["method"] == "op"
    assert result.metadata["min_size"] == 2
    assert np.isfinite(result.metadata["kernel_gamma"])
    assert result.metadata["kernel_metadata"]["approximation"] == "exact"


def test_kernel_callable_can_return_typed_gram_metadata() -> None:
    X = np.array([[0.0], [0.0], [5.0], [5.0]])

    def linear_with_metadata(data: np.ndarray) -> KernelMatrix:
        return KernelMatrix(gram=gram_linear(data), metadata={"name": "linear-explicit"})

    result = KernelCPD(
        penalty=0.1,
        kernel=linear_with_metadata,
        min_size=1,
        method="op",
        grid_jump=1,
    ).fit_predict(X)

    assert result.indices.tolist() == [2]
    assert result.metadata["kernel_metadata"]["name"] == "linear-explicit"


def test_linear_kernel_matches_explicit_euclidean_feature_oracle() -> None:
    X = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [3.0, 1.0], [3.1, 1.0]])
    K = gram_linear(X)
    prefix = build_kernel_prefix(K)

    expected_cps, expected_cost = _bruteforce_fixed_m(
        len(X),
        m=2,
        min_size=2,
        cost=lambda start, stop: _feature_cost(X, start, stop),
    )
    result = kcp_fixed_m(prefix, m=2, min_size=2)

    assert result.change_points.tolist() == expected_cps
    assert result.total_cost == pytest.approx(expected_cost)
    assert kernel_segment_cost(prefix, 0, 3) == pytest.approx(_feature_cost(X, 0, 3))


def test_exact_rbf_dp_matches_independent_bruteforce_kernel_objective() -> None:
    X = np.array([[0.0], [0.1], [0.2], [3.5], [3.6], [3.7]])
    K, _ = gram_rbf(X, gamma=0.5)
    prefix = build_kernel_prefix(K)

    expected_cps, expected_cost = _bruteforce_fixed_m(
        len(X),
        m=2,
        min_size=2,
        cost=lambda start, stop: _kernel_cost(K, start, stop),
    )
    result = kcp_fixed_m(prefix, m=2, min_size=2)

    assert result.change_points.tolist() == expected_cps
    assert result.total_cost == pytest.approx(expected_cost)


def test_rff_wrapper_converges_to_exact_rbf_segmentation() -> None:
    X = np.array([[0.0], [0.05], [0.1], [3.0], [3.05], [3.1]])
    exact = KernelCPD(
        penalty=0.05,
        kernel_kwargs={"gamma": 0.5},
        min_size=2,
        method="op",
    ).fit_predict(X)
    approx = KernelCPD(
        penalty=0.05,
        min_size=2,
        method="op",
        rff_config=RFFConfig(n_features=8192, gamma=0.5, seed=7),
    ).fit_predict(X)

    assert exact.indices.tolist() == [3]
    assert approx.indices.tolist() == exact.indices.tolist()
    assert approx.metadata["approximation"] == "rff"
    assert approx.metadata["kernel_metadata"]["rff_n_features"] == 8192
    assert approx.metadata["kernel_metadata"]["kernel_gamma"] == pytest.approx(0.5)
    assert approx.score == pytest.approx(exact.score, abs=0.15)


def test_constant_duplicate_inputs_have_stable_bandwidth_fallbacks() -> None:
    X = np.zeros((4, 2))
    K, gamma = gram_rbf(X)
    exact = kcp_penalized(build_kernel_prefix(K), penalty=0.1, min_size=1, method="op")
    rff = rbf_rff_map(X, RFFConfig(n_features=16, seed=0))

    assert gamma == pytest.approx(0.5)
    assert exact.change_points.tolist() == []
    assert rff.gamma == pytest.approx(0.5)


def test_kernel_prefix_rejects_invalid_or_too_large_grams() -> None:
    with pytest.raises(MemoryError):
        build_kernel_prefix(np.eye(4), max_bytes=8)
    with pytest.raises(ValueError, match="finite"):
        build_kernel_prefix(np.array([[1.0, np.nan], [np.nan, 1.0]]))
    with pytest.raises(ValueError, match="symmetric"):
        build_kernel_prefix(np.array([[1.0, 0.0], [0.5, 1.0]]))
    with pytest.raises(ValueError, match="positive semidefinite"):
        build_kernel_prefix(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_bandwidth_cv_changepoint_score_and_splits_are_deterministic() -> None:
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(0.0, 0.1, size=(8, 1)), rng.normal(3.0, 0.1, size=(8, 1))]
    cfg = BandwidthCVConfig(
        method="kfold",
        cv_folds=4,
        search_strategy="random",
        n_candidates=4,
        sigma_range=(0.1, 2.0),
        scoring="changepoint",
        seed=123,
    )

    split_a = _kfold_split(len(X), 4, seed=cfg.seed)
    split_b = _kfold_split(len(X), 4, seed=cfg.seed)
    sigmas_a = _generate_candidate_sigmas(cfg, X)
    sigmas_b = _generate_candidate_sigmas(cfg, X)
    score = _compute_changepoint_score(X[:8], X[8:], sigma=1.0)
    ts_train, ts_test = _time_series_split(len(X), 3)[0]

    assert [(a.tolist(), b.tolist()) for a, b in split_a] == [
        (a.tolist(), b.tolist()) for a, b in split_b
    ]
    assert np.array_equal(sigmas_a, sigmas_b)
    assert np.isfinite(score)
    assert ts_train.max() < ts_test.min()


def test_rff_prefix_and_result_retain_gamma_metadata() -> None:
    X = np.array([[0.0], [0.0], [5.0], [5.0]])
    rff = rbf_rff_map(X, RFFConfig(n_features=64, seed=0), gamma=0.5)
    prefix = build_feature_prefix(rff.Z)

    result = KernelCPD(
        penalty=0.1,
        min_size=1,
        method="op",
        rff_config=RFFConfig(n_features=64, gamma=0.5, seed=0),
    ).fit_predict(X)

    assert prefix.S.shape == (5, 64)
    assert result.metadata["kernel_gamma"] == pytest.approx(rff.gamma)
    assert result.metadata["kernel_metadata"]["rff_seed"] == 0
