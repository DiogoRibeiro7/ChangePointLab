"""Validate baseline fixtures against an installed ChangePointLab package.

Run this file from an environment where ChangePointLab has been installed either
editable or from a wheel. The script is intentionally located outside the package
root import path and uses only standard-library assertions plus NumPy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import changepoint_lab as cpl
from changepoint_lab import BOCPD, HSMM, KernelCPD, PELT
from changepoint_lab.algorithms.bayesian.bocpd import BOCPDConfig, ConstantHazard
from changepoint_lab.algorithms.optimization.pelt import NormalMeanKnownVar
from changepoint_lab.algorithms.state_space.emissions.gaussian_diag import (
    GaussianDiagParams,
    gaussian_diag_loglik,
)
from changepoint_lab.algorithms.state_space.hsmm import HSMMConfig, HSMMParams, PoissonDur


HERE = Path(__file__).resolve().parent


def _load_inputs() -> dict:
    return json.loads((HERE / "golden_inputs.json").read_text(encoding="utf-8"))["fixtures"]


def _load_expected() -> dict:
    return json.loads((HERE / "current_outputs.json").read_text(encoding="utf-8"))[
        "baselines"
    ]


def _round_list(values: np.ndarray, ndigits: int = 10) -> list[float]:
    return np.round(np.asarray(values, dtype=float), ndigits).tolist()


def main() -> None:
    inputs = _load_inputs()
    expected = _load_expected()

    x = np.asarray(inputs["pelt_gaussian_series"], dtype=float)
    pelt_res = PELT(cost_fn=NormalMeanKnownVar(sigma2=1.0), penalty=1.0, min_seg_len=2).fit_predict(x)
    assert pelt_res.indices.tolist() == expected["pelt_known_variance_oracle"]["change_points"]
    assert round(float(pelt_res.score), 10) == round(
        expected["pelt_known_variance_oracle"]["total_cost"], 10
    )

    stream = np.asarray(inputs["bocpd_binary_stream"], dtype=int)
    bocpd_res = BOCPD(
        ConstantHazard(mean_run_length=4),
        BOCPDConfig(max_run_length=8, prune_epsilon=0.0, cp_scale=1.0),
    ).run(stream)
    assert _round_list(bocpd_res.cp_prob) == expected["bocpd_beta_bernoulli_current"]["cp_prob"]
    assert bocpd_res.map_run_length.tolist() == expected["bocpd_beta_bernoulli_current"][
        "map_run_length"
    ]

    kernel_res = KernelCPD(penalty=0.1).fit_predict(
        np.asarray(inputs["kernel_points"], dtype=float)
    )
    assert type(kernel_res).__name__ == expected["kernel_cpd_current"]["wrapper_result_type"]
    assert kernel_res.indices.tolist() == expected["kernel_cpd_current"]["wrapper_indices"]

    obs = np.asarray(inputs["hsmm_observations"], dtype=float)
    emission_params = GaussianDiagParams(
        mu=np.array([[0.0], [1.0]]),
        var=np.array([[0.2], [0.2]]),
    )
    loglik = gaussian_diag_loglik(obs.reshape(-1, 1), emission_params)
    hsmm_params = HSMMParams(
        pi=np.array([1.0, 0.0]),
        A=np.array([[0.0, 1.0], [1.0, 0.0]]),
        duration=("poisson", PoissonDur(lam=np.array([2.0, 2.0]))),
    )
    hsmm = HSMM(
        HSMMConfig(K=2, Dmax=3, max_em_iters=1, learn_durations=False, seed=0),
        hsmm_params,
    )
    states, durations = hsmm.decode_viterbi(loglik)
    assert states.tolist() == expected["hsmm_core_oracle"]["states"]
    assert durations.tolist() == expected["hsmm_core_oracle"]["durations_by_end"]

    assert cpl.__version__ == "0.1.6"
    print("installed baseline ok")


if __name__ == "__main__":
    main()
