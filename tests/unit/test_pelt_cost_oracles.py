from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np
import pytest

from changepoint_lab.algorithms.optimization.pelt import (
    BetaBinomialCost,
    NormalMeanKnownVar,
    NormalMeanVarUnknown,
    aic_penalty,
    bic_penalty,
    pelt,
)


CostFormula = Callable[[np.ndarray, int, int], float]


def _normal_known_cost(sigma2: float) -> CostFormula:
    def cost(y: np.ndarray, a: int, b: int) -> float:
        segment = y[a:b]
        length = segment.size
        if length == 0:
            return float("inf")
        mean = float(np.mean(segment))
        sse = float(np.sum((segment - mean) ** 2))
        return (sse / sigma2) + length * math.log(2.0 * math.pi * sigma2)

    return cost


def _normal_unknown_cost(eps: float = 1e-12) -> CostFormula:
    def cost(y: np.ndarray, a: int, b: int) -> float:
        segment = y[a:b]
        length = segment.size
        if length <= 1:
            return float("inf")
        mean = float(np.mean(segment))
        sse = max(float(np.sum((segment - mean) ** 2)), eps)
        return length * (math.log(2.0 * math.pi) + math.log(sse / length) + 1.0)

    return cost


def _beta_binomial_cost(alpha: float = 1.0, beta: float = 1.0) -> CostFormula:
    def log_beta(a: float, b: float) -> float:
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    def cost(y: np.ndarray, a: int, b: int) -> float:
        segment = y[a:b]
        successes = float(np.sum(segment))
        failures = float(segment.size - successes)
        return -log_beta(successes + alpha, failures + beta) + log_beta(alpha, beta)

    return cost


def _all_segmentations(n: int) -> list[list[int]]:
    return [
        [idx + 1 for idx in range(n - 1) if mask & (1 << idx)]
        for mask in range(1 << max(0, n - 1))
    ]


def _oracle(
    y: Sequence[float],
    *,
    cost: CostFormula,
    penalty: float,
    min_seg_len: int,
) -> tuple[list[list[int]], float]:
    y_arr = np.asarray(y, dtype=float)
    best_score = float("inf")
    best_cps: list[list[int]] = []
    for cps in _all_segmentations(y_arr.size):
        edges = [0, *cps, y_arr.size]
        if any(b - a < min_seg_len for a, b in zip(edges[:-1], edges[1:], strict=True)):
            continue
        score = sum(cost(y_arr, a, b) for a, b in zip(edges[:-1], edges[1:], strict=True))
        score += penalty * len(cps)
        if score < best_score - 1e-10:
            best_score = score
            best_cps = [cps]
        elif math.isclose(score, best_score, rel_tol=1e-10, abs_tol=1e-10):
            best_cps.append(cps)
    assert best_cps
    return best_cps, best_score


@pytest.mark.parametrize(
    ("series", "cost_factory", "cost_formula", "min_seg_len", "penalty"),
    [
        (
            [-2.0, -1.0, 1.0, -1.0, -2.0],
            lambda: NormalMeanKnownVar(sigma2=1.0),
            _normal_known_cost(sigma2=1.0),
            2,
            1.0,
        ),
        (
            [0.0, 0.0, 0.1, 4.0, 4.1, 4.0],
            lambda: NormalMeanKnownVar(sigma2=1.0),
            _normal_known_cost(sigma2=1.0),
            2,
            1.0,
        ),
        (
            [0.0, -2.0, 1.0, -2.0, -2.0, 1.0],
            NormalMeanVarUnknown,
            _normal_unknown_cost(),
            2,
            0.5,
        ),
        (
            [0, 0, 0, 1, 1, 1, 1],
            BetaBinomialCost,
            _beta_binomial_cost(),
            2,
            0.25,
        ),
    ],
)
def test_pelt_matches_independent_oracle_boundaries_and_objective(
    series: Sequence[float],
    cost_factory: Callable[[], object],
    cost_formula: CostFormula,
    min_seg_len: int,
    penalty: float,
) -> None:
    expected_cps, expected_score = _oracle(
        series,
        cost=cost_formula,
        penalty=penalty,
        min_seg_len=min_seg_len,
    )
    result = pelt(series, cost_factory(), penalty=penalty, min_seg_len=min_seg_len)

    assert result.change_points in expected_cps
    assert result.total_cost == pytest.approx(expected_score)
    assert result.costs_per_segment.sum() + penalty * len(result.change_points) == pytest.approx(
        result.total_cost
    )


def test_pelt_exhaustive_binary_beta_binomial_objectives() -> None:
    for n in range(2, 8):
        for series in itertools.product([0.0, 1.0], repeat=n):
            for min_seg_len in range(1, min(3, n) + 1):
                expected_cps, expected_score = _oracle(
                    series,
                    cost=_beta_binomial_cost(),
                    penalty=0.5,
                    min_seg_len=min_seg_len,
                )
                result = pelt(series, BetaBinomialCost(), penalty=0.5, min_seg_len=min_seg_len)
                assert result.change_points in expected_cps
                assert result.total_cost == pytest.approx(expected_score)


def test_pelt_random_small_gaussian_objectives_match_oracle() -> None:
    rng = np.random.default_rng(20260723)
    cases = [
        (lambda: NormalMeanKnownVar(sigma2=2.0), _normal_known_cost(sigma2=2.0), 1),
        (NormalMeanVarUnknown, _normal_unknown_cost(), 2),
    ]
    for cost_factory, formula, min_allowed in cases:
        for n in range(3, 9):
            for _ in range(20):
                series = rng.normal(size=n)
                min_seg_len = int(rng.integers(min_allowed, min(4, n) + 1))
                penalty = float(rng.choice([0.1, 1.0, 3.0]))
                expected_cps, expected_score = _oracle(
                    series,
                    cost=formula,
                    penalty=penalty,
                    min_seg_len=min_seg_len,
                )
                result = pelt(
                    series,
                    cost_factory(),
                    penalty=penalty,
                    min_seg_len=min_seg_len,
                )
                assert result.change_points in expected_cps
                assert result.total_cost == pytest.approx(expected_score)


def test_pelt_rejects_nonfinite_and_multidimensional_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        pelt([0.0, math.nan, 1.0], NormalMeanKnownVar(sigma2=1.0), penalty=1.0)
    with pytest.raises(ValueError, match="finite"):
        pelt([0.0, math.inf, 1.0], NormalMeanKnownVar(sigma2=1.0), penalty=1.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        pelt(np.zeros((2, 2)), NormalMeanKnownVar(sigma2=1.0), penalty=1.0)


def test_cost_invariances_and_degenerate_segments() -> None:
    series = np.array([0.0, 0.1, -0.1, 4.0, 4.1, 3.9])
    shifted = series + 100.0
    known = pelt(series, NormalMeanKnownVar(sigma2=1.0), penalty=1.0, min_seg_len=2)
    known_shifted = pelt(shifted, NormalMeanKnownVar(sigma2=1.0), penalty=1.0, min_seg_len=2)
    assert known_shifted.change_points == known.change_points
    assert known_shifted.total_cost == pytest.approx(known.total_cost)

    unknown = NormalMeanVarUnknown()
    unknown.precompute(np.array([2.0, 2.0, 2.0]))
    assert math.isfinite(unknown.cost(0, 3))
    assert unknown.cost(0, 1) == math.inf

    binary = np.array([0, 0, 1, 1], dtype=float)
    complement = 1.0 - binary
    beta = pelt(binary, BetaBinomialCost(), penalty=0.5, min_seg_len=2)
    beta_complement = pelt(complement, BetaBinomialCost(), penalty=0.5, min_seg_len=2)
    assert beta_complement.change_points == beta.change_points
    assert beta_complement.total_cost == pytest.approx(beta.total_cost)


def test_aic_bic_helpers_are_deviance_scale_penalties() -> None:
    assert aic_penalty(params_per_segment=2) == pytest.approx(4.0)
    assert bic_penalty(params_per_segment=2, n=10) == pytest.approx(2.0 * math.log(10))
    assert bic_penalty(params_per_segment=2, n=1) == pytest.approx(2.0 * math.log(2))
