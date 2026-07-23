import itertools

import numpy as np
import pytest

from changepoint_lab import EDivisive
from changepoint_lab.algorithms.nonparametric.edivisive_core import (
    _best_split_statistic,
    _choose_block_size,
    _energy_stat_scan_from_ps,
    _pairwise_energy_dist_alpha,
    _prefix2d,
    _resample_block_permutation,
    _resample_circular_block_bootstrap,
    _resample_iid_permutation,
    edivisive,
)
from changepoint_lab.common.api_harmonizer import AlgorithmRegistry


def _manual_distance_alpha(X: np.ndarray, alpha: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i, j in itertools.product(range(n), repeat=2):
        D[i, j] = float(np.linalg.norm(X[i] - X[j]) ** alpha)
    return D


def _manual_energy_profile(D: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    out = np.empty(n - 1, dtype=float)
    for split in range(1, n):
        left = np.arange(split)
        right = np.arange(split, n)
        n_left = left.size
        n_right = right.size
        cross = float(D[np.ix_(left, right)].sum())
        within_left = float(D[np.ix_(left, left)].sum())
        within_right = float(D[np.ix_(right, right)].sum())
        out[split - 1] = (n_left * n_right) / n * (
            (2.0 * cross) / (n_left * n_right)
            - within_left / (n_left * n_left)
            - within_right / (n_right * n_right)
        )
    return out


def _hit_rate(
    *,
    delta: float,
    dimension: int,
    segment_len: int,
    reps: int,
    R: int,
    significance: float,
) -> float:
    hits = 0
    for seed in range(reps):
        rng = np.random.default_rng(4000 + seed)
        left = rng.normal(0.0, 1.0, size=(segment_len, dimension))
        right = rng.normal(delta, 1.0, size=(segment_len, dimension))
        x = np.vstack([left, right])
        result = edivisive(
            x,
            min_size=max(6, segment_len // 3),
            R=R,
            significance=significance,
            seed=seed,
            resample="iid",
        )
        tolerance = max(4, segment_len // 5)
        hits += any(abs(cp - segment_len) <= tolerance for cp in result.change_points)
    return hits / reps


def test_energy_statistic_matches_independent_multivariate_oracle() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 3.0], [5.0, 3.0]])
    D, tmp = _pairwise_energy_dist_alpha(X, alpha=1.0)
    assert tmp is None

    expected_D = _manual_distance_alpha(X, alpha=1.0)
    expected_profile = _manual_energy_profile(expected_D)
    profile, _, _ = _energy_stat_scan_from_ps(_prefix2d(D))

    assert D == pytest.approx(expected_D)
    assert profile == pytest.approx(expected_profile)


def test_alpha_two_uses_squared_euclidean_distances() -> None:
    X = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 4.0]])
    D, _ = _pairwise_energy_dist_alpha(X, alpha=2.0)

    assert D == pytest.approx(_manual_distance_alpha(X, alpha=2.0))
    assert D[0, 1] == pytest.approx(25.0)
    assert D[1, 2] == pytest.approx(9.0)


def test_best_split_includes_last_admissible_boundary() -> None:
    X = np.array([0.0, 0.0, 0.0, 0.0, 5.0, 5.0])
    D = _manual_distance_alpha(X, alpha=1.0)

    split, statistic, profile = _best_split_statistic(D, min_size=2)

    assert split == 4
    assert profile[3] == pytest.approx(statistic)
    assert np.isnan(profile[0])
    assert np.isnan(profile[-1])


def test_ties_choose_first_admissible_split() -> None:
    D = np.zeros((6, 6), dtype=float)

    split, statistic, profile = _best_split_statistic(D, min_size=2)

    assert split == 2
    assert statistic == pytest.approx(0.0)
    assert np.nan_to_num(profile, nan=-1.0).tolist() == [-1.0, 0.0, 0.0, 0.0, -1.0]


def test_resampling_paths_are_deterministic_and_shape_preserving() -> None:
    rng_a = np.random.default_rng(10)
    rng_b = np.random.default_rng(10)

    iid_a = _resample_iid_permutation(12, rng_a)
    iid_b = _resample_iid_permutation(12, rng_b)
    block = _resample_block_permutation(11, 3, np.random.default_rng(4))
    cbb = _resample_circular_block_bootstrap(11, 3, np.random.default_rng(4))

    assert np.array_equal(iid_a, iid_b)
    assert sorted(iid_a.tolist()) == list(range(12))
    assert sorted(block.tolist()) == list(range(11))
    assert cbb.shape == (11,)
    assert np.all((0 <= cbb) & (cbb < 11))
    assert _choose_block_size(100, None) >= 2
    with pytest.raises(ValueError, match="block_size"):
        _choose_block_size(10, 1)


def test_public_wrapper_exposes_low_level_resampling_controls() -> None:
    X = np.r_[np.zeros(12), np.ones(12)]
    wrapper = EDivisive(
        min_size=6,
        R=9,
        significance=0.2,
        max_cps=1,
        seed=5,
        progress=False,
        n_jobs=1,
        resample="block-permutation",
        block_size=3,
        chunk_size=4,
        use_memmap=True,
    ).fit_predict(X)

    provenance = wrapper.provenance
    assert provenance["significance"] == pytest.approx(0.2)
    assert provenance["max_cps"] == 1
    assert provenance["n_jobs"] == 1
    assert provenance["resample"] == "block-permutation"
    assert provenance["block_size"] == 3
    assert provenance["chunk_size"] == 4
    assert provenance["use_memmap"] is True
    assert wrapper.metadata["provenance"] == provenance


def test_api_harmonizer_edivisive_adapter_uses_keyword_controls() -> None:
    data = np.r_[np.zeros(8), np.ones(8)].reshape(-1, 1)
    registry = AlgorithmRegistry()

    result = registry.edivisive_adapter(
        data,
        min_size=4,
        R=9,
        significance=0.2,
        resample="iid",
        max_cps=1,
        n_jobs=1,
        seed=0,
    )

    assert result.model_name == "edivisive"
    assert result.parameters["resample"] == "iid"
    assert result.parameters["max_cps"] == 1
    assert isinstance(result.scores, list)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        edivisive([0.0, np.nan, 1.0, 2.0], min_size=2)
    with pytest.raises(ValueError, match="min_size"):
        edivisive([0.0, 1.0], min_size=0)
    with pytest.raises(ValueError, match="n_jobs"):
        edivisive(np.arange(8.0), min_size=2, n_jobs=2)


def test_iid_null_empirical_type_i_error_is_bounded() -> None:
    rejections = 0
    reps = 30
    for seed in range(reps):
        rng = np.random.default_rng(seed)
        result = edivisive(
            rng.normal(size=36),
            min_size=9,
            R=39,
            significance=0.1,
            seed=1000 + seed,
            resample="iid",
        )
        rejections += int(result.change_points.size > 0)

    assert rejections / reps <= 0.2


def test_block_resampling_null_paths_are_bounded_on_dependent_series() -> None:
    reps = 20
    for resample in ["block-permutation", "circular-block-bootstrap"]:
        rejections = 0
        for seed in range(reps):
            rng = np.random.default_rng(2000 + seed)
            eps = rng.normal(size=48)
            x = np.empty(48, dtype=float)
            x[0] = eps[0]
            for t in range(1, x.size):
                x[t] = 0.6 * x[t - 1] + eps[t]
            result = edivisive(
                x,
                min_size=8,
                R=29,
                significance=0.15,
                seed=seed,
                resample=resample,
                block_size=4,
            )
            rejections += int(result.change_points.size > 0)
        assert rejections / reps <= 0.4


def test_power_curve_increases_with_effect_size_dimension_and_segment_length() -> None:
    null_power = _hit_rate(delta=0.0, dimension=1, segment_len=24, reps=20, R=29, significance=0.15)
    moderate_power = _hit_rate(
        delta=0.75, dimension=1, segment_len=24, reps=20, R=29, significance=0.15
    )
    strong_power = _hit_rate(
        delta=1.5, dimension=1, segment_len=24, reps=20, R=29, significance=0.15
    )
    multivariate_power = _hit_rate(
        delta=1.5, dimension=3, segment_len=24, reps=20, R=29, significance=0.15
    )
    longer_power = _hit_rate(
        delta=1.5, dimension=1, segment_len=32, reps=20, R=29, significance=0.15
    )

    assert null_power <= 0.2
    assert moderate_power > null_power
    assert strong_power > moderate_power
    assert multivariate_power >= strong_power
    assert longer_power >= strong_power
