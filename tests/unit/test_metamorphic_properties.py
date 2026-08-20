from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from collections.abc import Sequence

import numpy as np
import pytest

from changepoint_lab import EDivisive, KernelCPD
from changepoint_lab.algorithms.bayesian.bocpd import (
    BOCPD,
    BOCPDConfig,
    BetaBernoulli,
    ConstantHazard,
)
from changepoint_lab.algorithms.bayesian.within_period import ModelPrior
from changepoint_lab.algorithms.bayesian.within_period.within_period_cpd import (
    Tau,
    WithinPeriodCore,
)
from changepoint_lab.algorithms.optimization.pelt import (
    BetaBinomialCost,
    NormalMeanKnownVar,
    pelt,
)
from changepoint_lab.algorithms.point_process.sliced_poisson import (
    SlicedPoissonCPD,
    SlicedPoissonConfig,
)
from changepoint_lab.algorithms.state_space.hsmm import (
    HSMM,
    HSMMConfig,
    HSMMParams,
    PoissonDur,
)
from changepoint_lab.algorithms.state_space.sdhmm import SDHMM, SDHMMConfig
from changepoint_lab.algorithms.state_space.sdhmm_mix_vi import (
    SDHMMMixVI,
    SDHMMMixVIConfig,
)
from changepoint_lab.core.segmentation import (
    CircularChangePoints,
    changepoints_from_labels,
    changepoints_to_edges,
    edges_to_changepoints,
    labels_from_changepoints,
)


def _all_partitions(n: int) -> list[list[int]]:
    return [
        [idx + 1 for idx in range(n - 1) if mask & (1 << idx)]
        for mask in range(1 << max(0, n - 1))
    ]


def _valid_for_min_segment_length(cps: Sequence[int], n: int, min_segment_length: int) -> bool:
    edges = [0, *cps, n]
    return all(
        stop - start >= min_segment_length
        for start, stop in zip(edges[:-1], edges[1:], strict=True)
    )


def _bocpd_cp_probability(data: np.ndarray, likelihood: BetaBernoulli) -> np.ndarray:
    model = BOCPD(
        ConstantHazard(mean_run_length=4),
        BOCPDConfig(max_run_length=12, prune_epsilon=0.0),
        likelihood=likelihood,
    )
    return model.run(data).cp_prob


def _shift_period_events(periods: Sequence[Sequence[float]], shift: float) -> list[tuple[float, ...]]:
    return [tuple(sorted((event + shift) % 1.0 for event in period)) for period in periods]


def _hsmm_model() -> HSMM:
    cfg = HSMMConfig(K=2, Dmax=4, max_em_iters=1, learn_durations=False, seed=0)
    params = HSMMParams(
        pi=np.array([1.0, 0.0]),
        A=np.array([[0.0, 1.0], [1.0, 0.0]]),
        duration=("poisson", PoissonDur(lam=np.array([3.0, 3.0]))),
    )
    return HSMM(cfg, params)


def test_boundary_edge_label_round_trips_are_bijections_for_valid_partitions() -> None:
    for n in range(2, 8):
        for min_segment_length in range(1, 4):
            for cps in _all_partitions(n):
                if not _valid_for_min_segment_length(cps, n, min_segment_length):
                    continue

                edges = changepoints_to_edges(cps, n=n, min_segment_length=min_segment_length)
                labels = labels_from_changepoints(n, cps, min_segment_length=min_segment_length)

                assert edges_to_changepoints(
                    edges,
                    n=n,
                    min_segment_length=min_segment_length,
                ).tolist() == cps
                assert changepoints_from_labels(labels).tolist() == cps
                assert labels.min(initial=0) == 0
                assert labels.max(initial=0) == len(cps)


def test_spawned_rngs_replay_in_fresh_processes_and_children_diverge() -> None:
    code = textwrap.dedent(
        """
        import json
        from changepoint_lab.core.random import spawn_rngs

        streams = spawn_rngs(20260820, 3)
        draws = [stream.integers(0, 1_000_000, size=8).tolist() for stream in streams]
        print(json.dumps(draws))
        """
    )
    first = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    second = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    draws = json.loads(first.stdout)
    assert first.stdout == second.stdout
    assert len({tuple(child) for child in draws}) == 3


def test_pelt_gaussian_affine_invariance_uses_matching_variance_units() -> None:
    x = np.array([0.0, 0.1, -0.1, 4.0, 4.1, 3.9])
    base = pelt(x, NormalMeanKnownVar(sigma2=1.0), penalty=1.0, min_seg_len=2)
    shifted = pelt(x + 100.0, NormalMeanKnownVar(sigma2=1.0), penalty=1.0, min_seg_len=2)
    scaled = pelt(3.0 * x + 100.0, NormalMeanKnownVar(sigma2=9.0), penalty=1.0, min_seg_len=2)

    assert shifted.change_points == base.change_points
    assert scaled.change_points == base.change_points

    small = np.array([0.0, 0.0, 1.0, 1.0])
    unscaled_units = pelt(small, NormalMeanKnownVar(sigma2=1.0), penalty=1.0, min_seg_len=2)
    changed_units = pelt(10.0 * small, NormalMeanKnownVar(sigma2=1.0), penalty=1.0, min_seg_len=2)
    assert unscaled_units.change_points == []
    assert changed_units.change_points == [2]


def test_beta_binomial_pelt_is_invariant_to_binary_label_complement() -> None:
    x = np.array([0, 0, 0, 1, 1, 1, 0, 0], dtype=float)
    base = pelt(x, BetaBinomialCost(), penalty=0.5, min_seg_len=2)
    complemented = pelt(1.0 - x, BetaBinomialCost(), penalty=0.5, min_seg_len=2)

    assert complemented.change_points == base.change_points
    assert complemented.total_cost == pytest.approx(base.total_cost)


def test_bocpd_complement_symmetry_requires_symmetric_beta_prior() -> None:
    data = np.array([0, 0, 0, 1, 1, 1, 0, 1], dtype=int)
    complemented = 1 - data

    symmetric = _bocpd_cp_probability(data, BetaBernoulli(1.0, 1.0))
    symmetric_complement = _bocpd_cp_probability(complemented, BetaBernoulli(1.0, 1.0))
    asymmetric = _bocpd_cp_probability(data, BetaBernoulli(2.0, 5.0))
    asymmetric_complement = _bocpd_cp_probability(complemented, BetaBernoulli(2.0, 5.0))

    assert symmetric_complement == pytest.approx(symmetric)
    assert not np.allclose(asymmetric_complement, asymmetric)


def test_edivisive_energy_distance_is_translation_and_positive_scale_invariant() -> None:
    x = np.r_[np.zeros(8), np.ones(8) * 5.0]
    detector = EDivisive(min_size=4, R=19, seed=0, significance=0.1, max_cps=1)

    base = detector.fit_predict(x)
    shifted = EDivisive(min_size=4, R=19, seed=0, significance=0.1, max_cps=1).fit_predict(
        x + 20.0
    )
    scaled = EDivisive(min_size=4, R=19, seed=0, significance=0.1, max_cps=1).fit_predict(
        3.0 * x - 7.0
    )

    assert shifted.indices.tolist() == base.indices.tolist() == [8]
    assert scaled.indices.tolist() == base.indices.tolist()


def test_rbf_kernel_cpd_preserves_boundaries_under_distance_preserving_transforms() -> None:
    x = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [4.0, 4.0], [4.1, 4.0], [4.0, 4.1]]
    )
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    transformed = (x @ rotation.T) + np.array([10.0, -3.0])
    scaled = 2.5 * x + np.array([1.0, -4.0])

    def fit(data: np.ndarray) -> list[int]:
        return KernelCPD(penalty=0.1, min_size=2, method="op").fit_predict(data).indices.tolist()

    assert fit(x) == [3]
    assert fit(transformed) == fit(x)
    assert fit(scaled) == fit(x)


def test_within_period_log_posterior_rotates_with_periodic_data_and_boundaries() -> None:
    x = np.array([0, 1, 1, 0, 0, 1] * 3, dtype=bool)
    tau: Tau = (0, 3)
    shift = 2

    model = WithinPeriodCore(ModelPrior(N=6, l=2))
    model._prepare_counts(x)
    rotated = WithinPeriodCore(ModelPrior(N=6, l=2))
    rotated._prepare_counts(np.roll(x.reshape(-1, 6), shift, axis=1).ravel())
    rotated_tau = tuple(sorted((boundary + shift) % 6 for boundary in tau))

    assert CircularChangePoints(period=6, indices=np.array(tau)).rotated(shift).indices.tolist() == [
        *rotated_tau
    ]
    assert rotated._log_posterior_tau(rotated_tau) == pytest.approx(
        model._log_posterior_tau(tau)
    )


def test_sliced_poisson_constant_intensity_segmentation_ignores_event_phase() -> None:
    periods = [(0.1,)] * 6 + [(0.1, 0.2, 0.3, 0.4, 0.5)] * 6
    shifted = _shift_period_events(periods, 0.37)
    config = SlicedPoissonConfig(
        period=1.0,
        n_basis=1,
        degree=0,
        min_segment_periods=3,
        penalty=1.0,
    )

    base = SlicedPoissonCPD(config).fit_predict(periods)
    phase_shifted = SlicedPoissonCPD(config).fit_predict(shifted)

    assert base.change_points == [6]
    assert phase_shifted.change_points == base.change_points
    assert phase_shifted.total_cost == pytest.approx(base.total_cost)


def test_hsmm_decoding_is_invariant_to_per_time_loglik_offsets() -> None:
    loglik = np.array(
        [
            [0.0, -8.0],
            [0.0, -8.0],
            [0.0, -8.0],
            [-8.0, 0.0],
            [-8.0, 0.0],
            [-8.0, 0.0],
        ]
    )
    offsets = np.linspace(-3.0, 3.0, loglik.shape[0])[:, None]

    base = _hsmm_model().fit_predict(loglik)
    shifted = _hsmm_model().fit_predict(loglik + offsets)

    assert shifted.indices.tolist() == base.indices.tolist()
    assert shifted.states.tolist() == base.states.tolist()
    assert shifted.segment_durations.tolist() == base.segment_durations.tolist()


def test_scaled_dirichlet_hmm_wrappers_are_invariant_to_row_scale() -> None:
    x = np.array(
        [
            [0.90, 0.10],
            [0.85, 0.15],
            [0.20, 0.80],
            [0.15, 0.85],
            [0.90, 0.10],
            [0.85, 0.15],
        ]
    )
    row_scaled = x * np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])

    cfg = SDHMMConfig(K=2, max_iter=2, min_iter=1, seed=7, em_steps=1, lr_alpha=0.001, lr_beta=0.001)
    base = SDHMM(cfg).fit_predict(x)
    scaled = SDHMM(cfg).fit_predict(row_scaled)
    assert scaled.states.tolist() == base.states.tolist()

    mix_cfg = SDHMMMixVIConfig(
        K=2,
        M=1,
        max_iter=2,
        min_iter=1,
        seed=7,
        em_steps=1,
        lr_alpha=0.001,
        lr_beta=0.001,
    )
    mix_base = SDHMMMixVI(mix_cfg).fit_predict(x)
    mix_scaled = SDHMMMixVI(mix_cfg).fit_predict(row_scaled)
    assert mix_scaled.states.tolist() == mix_base.states.tolist()
