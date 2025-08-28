# tests/test_bayesian_blocks.py

from __future__ import annotations
import numpy as np
import pytest

from bayesian_blocks import (
    bayesian_blocks_events,
    bayesian_blocks_counts,
    bayesian_blocks_bernoulli,
)
from bb_utils import blocks_to_labels_index

rng = np.random.default_rng(0)

def test_counts_single_block_when_constant_rate():
    N = 300
    rate = 2.5
    counts = rng.poisson(rate, size=N)
    res = bayesian_blocks_counts(counts, p0=0.01)  # strong penalty
    # should usually prefer a single block
    assert len(res.block_value) == 1
    yhat = blocks_to_labels_index(N, res)
    assert yhat.shape == (N,)
    # MLE near global mean
    assert abs(res.block_value[0] - counts.mean()) < 0.5

def test_counts_two_blocks_clear_jump():
    N1, N2 = 120, 180
    c1 = rng.poisson(4.0, size=N1)
    c2 = rng.poisson(1.0, size=N2)
    counts = np.r_[c1, c2]
    res = bayesian_blocks_counts(counts, p0=0.05)
    # allow a small margin (BB can place edge near the true boundary)
    assert len(res.block_value) in (2, 3)
    # rates should be ordered as high then low on average
    assert res.block_value[0] > res.block_value[-1]

def test_bernoulli_change_in_probability():
    N1, N2 = 150, 150
    x1 = rng.binomial(1, 0.2, size=N1)
    x2 = rng.binomial(1, 0.7, size=N2)
    x = np.r_[x1, x2]
    res = bayesian_blocks_bernoulli(successes=x, trials=None, p0=0.05)
    assert len(res.block_value) >= 2
    assert res.block_value[0] < res.block_value[-1]

def test_events_piecewise_rates():
    # build two-rate Poisson process on [0, 10]
    # rate 3 up to 4.5, then rate 0.8
    t1 = np.cumsum(rng.exponential(1/3.0, size=200))
    t1 = t1[t1 < 4.5]
    t2 = 4.5 + np.cumsum(rng.exponential(1/0.8, size=300))
    t2 = t2[t2 < 10.0]
    t = np.sort(np.r_[t1, t2])
    res = bayesian_blocks_events(t, t_start=0.0, t_stop=10.0, p0=0.05)
    # expect ≥2 blocks and a drop in rate
    assert len(res.block_value) >= 2
    assert res.block_value[0] > res.block_value[-1]
