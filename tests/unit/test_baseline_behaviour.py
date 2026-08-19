from __future__ import annotations

import itertools
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import changepoint_lab as cpl
from changepoint_lab import BOCPD, EDivisive, HSMM, KernelCPD, PELT, SDHMM, SDHMMMixVI
from changepoint_lab.algorithms.bayesian.bocpd import BOCPDConfig, ConstantHazard
from changepoint_lab.algorithms.bayesian.within_period import ModelPrior, RJConfig, WithinPeriodCPD
from changepoint_lab.algorithms.bayesian.within_period.within_period_cpd import _is_valid_tau
from changepoint_lab.algorithms.kernel import kcp_core as kcp
from changepoint_lab.algorithms.kernel.kcp_rff import (
    RFFConfig,
    build_feature_prefix,
    rbf_rff_map,
    rff_kcp_penalized,
)
from changepoint_lab.algorithms.optimization.pelt import (
    NormalMeanKnownVar,
    NormalMeanVarUnknown,
    pelt,
    pelt_concave_penalty,
)
from changepoint_lab.algorithms.state_space.emissions.gaussian_diag import (
    GaussianDiagParams,
    gaussian_diag_loglik,
)
from changepoint_lab.algorithms.state_space.hsmm import HSMMConfig, HSMMParams, PoissonDur


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "baseline"


def _load_inputs() -> dict:
    return json.loads((FIXTURE_DIR / "golden_inputs.json").read_text(encoding="utf-8"))[
        "fixtures"
    ]


def _load_expected() -> dict:
    return json.loads((FIXTURE_DIR / "current_outputs.json").read_text(encoding="utf-8"))[
        "baselines"
    ]


def _round_list(values: np.ndarray | list[float], ndigits: int = 10) -> list[float]:
    return np.round(np.asarray(values, dtype=float), ndigits).tolist()


def _normal_known_cost(y: np.ndarray, a: int, b: int, sigma2: float) -> float:
    segment = y[a:b]
    length = segment.size
    mean = float(np.mean(segment))
    sse = float(np.sum((segment - mean) ** 2))
    return (sse / sigma2) + length * math.log(2.0 * math.pi * sigma2)


def _bruteforce_penalized_segments(
    y: np.ndarray,
    *,
    penalty: float,
    min_size: int,
    cost,
) -> tuple[list[int], float]:
    n = y.size
    best_cps: list[int] | None = None
    best_score = float("inf")
    for mask in range(1 << (n - 1)):
        cps = [idx + 1 for idx in range(n - 1) if mask & (1 << idx)]
        edges = [0, *cps, n]
        if any(b - a < min_size for a, b in zip(edges[:-1], edges[1:], strict=True)):
            continue
        score = sum(cost(y, a, b) for a, b in zip(edges[:-1], edges[1:], strict=True))
        score += penalty * len(cps)
        if score < best_score:
            best_score = score
            best_cps = cps
    assert best_cps is not None
    return best_cps, best_score


def _kernel_cost_from_matrix(K: np.ndarray, a: int, b: int) -> float:
    block = K[a:b, a:b]
    return float(np.trace(block) - np.sum(block) / (b - a))


def _bruteforce_kernel_fixed_m(K: np.ndarray, *, m: int, min_size: int) -> tuple[list[int], float]:
    n = K.shape[0]
    best_cps: list[int] | None = None
    best_score = float("inf")
    for cps in itertools.combinations(range(1, n), m - 1):
        edges = [0, *cps, n]
        if any(b - a < min_size for a, b in zip(edges[:-1], edges[1:], strict=True)):
            continue
        score = sum(_kernel_cost_from_matrix(K, a, b) for a, b in zip(edges[:-1], edges[1:], strict=True))
        if score < best_score:
            best_score = score
            best_cps = list(cps)
    assert best_cps is not None
    return best_cps, best_score


def _labels_from_edges(n: int, edges: list[int]) -> list[int]:
    labels = [0] * n
    for label, (a, b) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        for idx in range(a, b):
            labels[idx] = label
    return labels


def _circular_lengths(tau: tuple[int, ...], N: int) -> list[int]:
    if not tau:
        return [N]
    lengths: list[int] = []
    prev = tau[-1]
    for cp in tau:
        distance = (cp - prev) % N
        lengths.append(N if distance == 0 else distance)
        prev = cp
    return lengths


def _valid_circular_taus(N: int, min_segment_length: int, change_count: int) -> list[list[int]]:
    if change_count == 0:
        return [[]] if N >= min_segment_length else []
    valid: list[list[int]] = []
    for tau in itertools.combinations(range(N), change_count + 1):
        if all(length >= min_segment_length for length in _circular_lengths(tau, N)):
            valid.append(list(tau))
    return valid


def _hsmm_log_duration_table(lam: float, Dmax: int) -> list[float]:
    log_base = []
    for d in range(1, Dmax + 1):
        log_base.append(-lam + d * math.log(lam) - math.lgamma(d + 1.0))
    log_z = math.log(sum(math.exp(v) for v in log_base))
    return [v - log_z for v in log_base]


def _hsmm_fixture_loglik() -> np.ndarray:
    inputs = _load_inputs()
    obs = np.asarray(inputs["hsmm_observations"], dtype=float)
    params = GaussianDiagParams(mu=np.array([[0.0], [1.0]]), var=np.array([[0.2], [0.2]]))
    return gaussian_diag_loglik(obs.reshape(-1, 1), params)


def _fresh_process_baseline_json() -> str:
    code = r"""
import json
import numpy as np
from changepoint_lab import BOCPD, PELT
from changepoint_lab.algorithms.bayesian.bocpd import BOCPDConfig, ConstantHazard
from changepoint_lab.algorithms.optimization.pelt import NormalMeanKnownVar

x = np.array([0.0, 0.0, 0.0, 4.0, 4.0, 4.0])
pelt_res = PELT(cost_fn=NormalMeanKnownVar(sigma2=1.0), penalty=1.0, min_seg_len=2).fit_predict(x)
stream = np.array([0, 0, 0, 1, 1, 1], dtype=int)
bocpd_res = BOCPD(
    ConstantHazard(mean_run_length=4),
    BOCPDConfig(max_run_length=8, prune_epsilon=0.0, cp_scale=1.0),
).run(stream)
print(json.dumps({
    "pelt_indices": pelt_res.indices.tolist(),
    "pelt_score": round(float(pelt_res.score), 10),
    "bocpd_cp_prob": np.round(bocpd_res.cp_prob, 10).tolist(),
    "bocpd_map_run_length": bocpd_res.map_run_length.tolist(),
}, sort_keys=True))
"""
    return subprocess.check_output([sys.executable, "-c", code], text=True).strip()


def test_public_wrappers_match_recorded_baselines() -> None:
    inputs = _load_inputs()
    expected = _load_expected()

    x = np.asarray(inputs["pelt_gaussian_series"], dtype=float)
    pelt_expected = expected["pelt_known_variance_oracle"]
    pelt_res = PELT(cost_fn=NormalMeanKnownVar(sigma2=1.0), penalty=1.0, min_seg_len=2).fit_predict(x)
    assert pelt_res.indices.tolist() == pelt_expected["change_points"]
    assert pelt_res.metadata["labels"].tolist() == pelt_expected["labels"]
    assert sorted(pelt_res.metadata) == pelt_expected["metadata_keys"]
    assert pelt_res.score == pytest.approx(pelt_expected["total_cost"])

    stream = np.asarray(inputs["bocpd_binary_stream"], dtype=int)
    bocpd_expected = expected["bocpd_beta_bernoulli_current"]
    bocpd = BOCPD(
        ConstantHazard(mean_run_length=4),
        BOCPDConfig(max_run_length=8, prune_epsilon=0.0, cp_scale=1.0),
    )
    bocpd_res = bocpd.run(stream)
    assert _round_list(bocpd_res.cp_prob) == bocpd_expected["cp_prob"]
    assert bocpd_res.map_run_length.tolist() == bocpd_expected["map_run_length"]
    assert _round_list(bocpd_res.pred_mean) == bocpd_expected["pred_mean"]
    assert list(bocpd_res.run_length_posterior.shape) == bocpd_expected["run_length_posterior_shape"]
    bocpd_wrapper = BOCPD(
        ConstantHazard(mean_run_length=4),
        BOCPDConfig(max_run_length=8, prune_epsilon=0.0, cp_scale=1.0),
    ).fit_predict(stream)
    assert bocpd_wrapper.indices.tolist() == bocpd_expected["wrapper_indices"]
    assert sorted(bocpd_wrapper.metadata) == bocpd_expected["metadata_keys"]

    ediv_expected = expected["edivisive_current"]
    ediv = EDivisive(min_size=2, R=9, seed=0).fit_predict(
        np.asarray(inputs["edivisive_series"], dtype=float)
    )
    assert ediv.indices.tolist() == ediv_expected["indices"]
    assert sorted(ediv.metadata) == ediv_expected["metadata_keys"]
    assert len(ediv.metadata["splits"]) == ediv_expected["split_count"]


def test_low_level_entry_points_match_recorded_baselines() -> None:
    inputs = _load_inputs()
    expected = _load_expected()

    x = np.asarray(inputs["pelt_gaussian_series"], dtype=float)
    pelt_unknown = pelt(x, NormalMeanVarUnknown(), penalty=1.0, min_seg_len=2)
    unknown_expected = expected["pelt_unknown_variance_current"]
    assert pelt_unknown.change_points == unknown_expected["change_points"]
    assert pelt_unknown.labels.tolist() == unknown_expected["labels"]
    assert pelt_unknown.total_cost == pytest.approx(unknown_expected["total_cost"])

    pelt_concave = pelt_concave_penalty(
        x,
        NormalMeanVarUnknown(),
        f=lambda m: float(m),
        fprime=lambda m: 1.0,
        min_seg_len=2,
        max_iter=5,
    )
    assert pelt_concave.change_points == expected["pelt_concave_current"]["change_points"]
    assert pelt_concave.total_cost == pytest.approx(
        expected["pelt_concave_current"]["total_cost"]
    )

    X = np.asarray(inputs["kernel_points"], dtype=float)
    pref = kcp.build_kernel_prefix(kcp.gram_linear(X))
    kcp_current = expected["kernel_cpd_current"]
    kcp_res = kcp.kcp_penalized(pref, penalty=0.1, min_size=1, method="op")
    assert kcp_res.change_points.tolist() == kcp_current["change_points"]
    assert kcp_res.edges.tolist() == kcp_current["edges"]
    assert kcp_res.labels.tolist() == kcp_current["labels"]
    assert kcp_res.total_cost == pytest.approx(kcp_current["total_cost"])
    assert _round_list(kcp_res.costs_per_segment) == kcp_current["costs_per_segment"]

    rff = rbf_rff_map(X, RFFConfig(n_features=4, seed=0), gamma=0.5)
    rff_res = rff_kcp_penalized(
        build_feature_prefix(rff.Z), gamma_pen=0.1, min_size=1, method="op"
    )
    rff_expected = expected["rff_kernel_current"]
    assert rff.gamma == pytest.approx(rff_expected["gamma"])
    assert list(rff.Z.shape) == rff_expected["feature_shape"]
    assert rff_res.change_points.tolist() == rff_expected["change_points"]
    assert rff_res.edges.tolist() == rff_expected["edges"]
    assert rff_res.labels.tolist() == rff_expected["labels"]
    assert rff_res.total_cost == pytest.approx(rff_expected["total_cost"])


def test_scientific_oracles_are_independent_of_package_dp_paths() -> None:
    inputs = _load_inputs()
    expected = _load_expected()

    x = np.asarray(inputs["pelt_gaussian_series"], dtype=float)
    pelt_expected = expected["pelt_known_variance_oracle"]
    cps, score = _bruteforce_penalized_segments(
        x,
        penalty=1.0,
        min_size=2,
        cost=lambda y, a, b: _normal_known_cost(y, a, b, sigma2=1.0),
    )
    assert cps == pelt_expected["change_points"]
    assert score == pytest.approx(pelt_expected["total_cost"])

    X = np.asarray(inputs["kernel_points"], dtype=float)
    K = X @ X.T
    kernel_expected = expected["kernel_cpd_oracle"]
    kernel_cps, kernel_score = _bruteforce_kernel_fixed_m(K, m=2, min_size=1)
    assert kernel_cps == kernel_expected["change_points"]
    assert kernel_score == pytest.approx(kernel_expected["fixed_m_total_cost"])
    assert _labels_from_edges(len(X), kernel_expected["edges"]) == kernel_expected["labels"]

    circular_expected = expected["circular_segmentation_oracle"]
    assert _valid_circular_taus(6, 2, 0) == circular_expected["valid_tau_by_change_count"]["0"]
    valid_one = _valid_circular_taus(6, 2, 1)
    assert sorted(valid_one) == sorted(circular_expected["valid_tau_by_change_count"]["1"])
    for tau in itertools.combinations(range(6), 2):
        assert _is_valid_tau(tau, 6, 2) == (list(tau) in valid_one)

    hsmm_expected = expected["hsmm_core_oracle"]
    duration_logp = _hsmm_log_duration_table(lam=2.0, Dmax=3)
    assert _round_list(duration_logp) == hsmm_expected["log_duration_table"][0]


def test_state_space_and_periodic_baselines() -> None:
    inputs = _load_inputs()
    expected = _load_expected()

    wp = WithinPeriodCPD(ModelPrior(N=20, l=5))
    wp.fit(
        np.asarray(inputs["within_period_binary_stream"], dtype=bool),
        cfg=RJConfig(iters=20, burn=5, thin=5, seed=0),
    )
    wp_expected = expected["within_period_current"]
    assert list(wp.result.mode_tau) == wp_expected["mode_tau"]
    assert len(wp.result.samples_tau) == wp_expected["sample_count"]
    assert wp.result.changepoint_hist.tolist() == wp_expected["changepoint_hist"]
    assert _round_list(wp.result.log_posteriors) == wp_expected["log_posteriors"]

    L = _hsmm_fixture_loglik()
    params = HSMMParams(
        pi=np.array([1.0, 0.0]),
        A=np.array([[0.0, 1.0], [1.0, 0.0]]),
        duration=("poisson", PoissonDur(lam=np.array([2.0, 2.0]))),
    )
    hsmm = HSMM(
        HSMMConfig(K=2, Dmax=3, max_em_iters=1, learn_durations=False, seed=0),
        params,
    )
    states, durations = hsmm.decode_viterbi(L)
    hsmm_expected = expected["hsmm_core_oracle"]
    assert states.tolist() == hsmm_expected["states"]
    assert durations.tolist() == hsmm_expected["durations_by_end"]
    assert _round_list(hsmm._log_dur_table(4)) == hsmm_expected["log_duration_table"]

    wrapper_params = HSMMParams(
        pi=np.array([1.0, 0.0]),
        A=np.array([[0.0, 1.0], [1.0, 0.0]]),
        duration=("poisson", PoissonDur(lam=np.array([2.0, 2.0]))),
    )
    hsmm_wrapper = HSMM(
        HSMMConfig(K=2, Dmax=3, max_em_iters=1, learn_durations=False, seed=0),
        wrapper_params,
    ).fit_predict(L)
    wrapper_expected = expected["hsmm_wrapper_current"]
    assert type(hsmm_wrapper).__name__ == wrapper_expected["result_type"]
    assert hsmm_wrapper.indices.tolist() == wrapper_expected["indices"]
    assert hsmm_wrapper.states.tolist() == wrapper_expected["states"]
    assert hsmm_wrapper.segment_durations.tolist() == wrapper_expected["durations_by_end"]
    assert sorted(hsmm_wrapper.metadata) == wrapper_expected["metadata_keys"]

    compositions = np.asarray(inputs["sdhmm_compositions"], dtype=float)
    sdhmm = SDHMM(cpl.SDHMMConfig(K=2, max_iter=2, min_iter=1, seed=0)).fit_predict(
        compositions
    )
    sdhmm_expected = expected["sdhmm_current"]
    assert sdhmm.indices.tolist() == sdhmm_expected["indices"]
    assert sdhmm.metadata["states"].tolist() == sdhmm_expected["states"]
    assert sorted(sdhmm.metadata) == sdhmm_expected["metadata_keys"]


def test_documented_broken_paths_raise_recorded_exceptions() -> None:
    inputs = _load_inputs()
    expected = _load_expected()

    kernel_expected = expected["kernel_cpd_current"]
    kernel_wrapper = KernelCPD(penalty=0.1).fit_predict(
        np.asarray(inputs["kernel_points"], dtype=float)
    )
    assert type(kernel_wrapper).__name__ == kernel_expected["wrapper_result_type"]
    assert kernel_wrapper.indices.tolist() == kernel_expected["wrapper_indices"]
    assert sorted(kernel_wrapper.metadata) == kernel_expected["wrapper_metadata_keys"]

    wp_expected = expected["within_period_tiny_current"]
    tiny = WithinPeriodCPD(ModelPrior(N=4, l=1)).fit(
        np.asarray(inputs["within_period_tiny_stream"], dtype=bool),
        cfg=RJConfig(iters=40, burn=10, thin=5, seed=0),
    )
    assert list(tiny.result.mode_tau) == wp_expected["mode_tau"]
    assert len(tiny.result.samples_tau) == wp_expected["sample_count"]
    assert tiny.result.changepoint_hist.tolist() == wp_expected["changepoint_hist"]
    assert _round_list(tiny.result.log_posteriors) == wp_expected["log_posteriors"]

    mix_expected = expected["sdhmm_mix_current"]
    mix_result = SDHMMMixVI(
        cpl.SDHMMMixVIConfig(K=2, M=1, max_iter=2, min_iter=1, seed=0)
    ).fit_predict(np.asarray(inputs["sdhmm_compositions"], dtype=float))
    assert type(mix_result).__name__ == mix_expected["result_type"]
    assert mix_result.indices.tolist() == mix_expected["indices"]
    assert mix_result.states.tolist() == mix_expected["states"]
    assert sorted(mix_result.metadata) == mix_expected["metadata_keys"]

    legacy_expected = expected["legacy_current"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from changepoint_lab import pelt as legacy_pelt

        legacy_pelt(
            np.asarray(inputs["pelt_gaussian_series"], dtype=float),
            cost_fn=NormalMeanVarUnknown(),
            penalty=1.0,
            min_seg_len=2,
        )
    assert any(
        item.category.__name__ == legacy_expected["pelt_warning_category"]
        and str(item.message) == legacy_expected["pelt_warning_message"]
        for item in caught
    )

    with pytest.raises(AttributeError) as legacy_error:
        cpl.bocpd
    assert type(legacy_error.value).__name__ == legacy_expected["bocpd_exception_type"]
    assert str(legacy_error.value) == legacy_expected["bocpd_exception_message"]


def test_fresh_process_repetition_is_deterministic() -> None:
    first = _fresh_process_baseline_json()
    second = _fresh_process_baseline_json()
    assert first == second

    expected = _load_expected()
    payload = json.loads(first)
    assert payload["pelt_indices"] == expected["pelt_known_variance_oracle"]["change_points"]
    assert payload["pelt_score"] == pytest.approx(
        expected["pelt_known_variance_oracle"]["total_cost"]
    )
    assert payload["bocpd_cp_prob"] == expected["bocpd_beta_bernoulli_current"]["cp_prob"]
    assert payload["bocpd_map_run_length"] == expected["bocpd_beta_bernoulli_current"][
        "map_run_length"
    ]


def test_golden_fixtures_are_path_time_and_platform_neutral() -> None:
    for path in FIXTURE_DIR.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert str(ROOT) not in text
        assert "\\" not in text
        payload = json.loads(text)
        assert payload["created_on"] == "2026-07-23"
