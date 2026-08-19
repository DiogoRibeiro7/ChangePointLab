from __future__ import annotations

import numpy as np
import pytest

from changepoint_lab import WithinPeriodCPD
from changepoint_lab.algorithms.bayesian.bocpd import PoissonGamma
from changepoint_lab.algorithms.bayesian.within_period import ModelPrior
from changepoint_lab.core.validation import (
    as_binary_array,
    as_count_array,
    as_probability_array,
    as_square_matrix,
    as_strictly_increasing_times,
)


def test_binary_validator_accepts_bool_and_exact_zero_one() -> None:
    assert as_binary_array(np.array([True, False])).tolist() == [True, False]
    assert as_binary_array(np.array([0, 1])).tolist() == [False, True]


@pytest.mark.parametrize("values", [np.array([0, 2]), np.array([0.0, 0.5])])
def test_binary_validator_rejects_non_binary_inputs(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="binary"):
        as_binary_array(values)


def test_binary_validator_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="empty"):
        as_binary_array(np.array([]))


def test_count_validator_rejects_bool_negative_and_fractional_counts() -> None:
    with pytest.raises(ValueError, match="not bool"):
        as_count_array(np.array([True, False]))
    with pytest.raises(ValueError, match="integer counts"):
        as_count_array(np.array([1, -1]))
    with pytest.raises(ValueError, match="integer counts"):
        as_count_array(np.array([1.5]))


def test_probability_time_and_matrix_validators_cover_public_domains() -> None:
    assert as_probability_array(np.array([0.0, 0.5, 1.0])).shape == (3,)
    with pytest.raises(ValueError, match="probabilities"):
        as_probability_array(np.array([1.2]))

    assert as_strictly_increasing_times(np.array([0.0, 1.0, 3.0])).tolist() == [0.0, 1.0, 3.0]
    with pytest.raises(ValueError, match="strictly increasing"):
        as_strictly_increasing_times(np.array([0.0, 0.0, 1.0]))

    assert as_square_matrix(np.eye(2), name="K", psd=True).shape == (2, 2)
    with pytest.raises(ValueError, match="square"):
        as_square_matrix(np.ones((2, 3)), name="K")
    with pytest.raises(ValueError, match="positive semidefinite"):
        as_square_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]), name="K", psd=True)


def test_bocpd_likelihoods_reject_ambiguous_observation_domains() -> None:
    poisson = PoissonGamma()
    poisson.init_stats(2)
    with pytest.raises(ValueError, match="integer counts"):
        poisson.predictive_prob(True)
    with pytest.raises(ValueError, match="integer counts"):
        poisson.predictive_prob(1.5)


def test_within_period_wrapper_rejects_non_binary_series() -> None:
    detector = WithinPeriodCPD(ModelPrior(N=4, l=1, gamma=1.0, pois_lambda=1.0))
    with pytest.raises(ValueError, match="binary"):
        detector.fit(np.array([0, 1, 2, 0]))
