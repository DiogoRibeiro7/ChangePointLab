"""Quick start examples for changepoint-toolkit.

Demonstrates basic usage of BOCPD, Within-Period CPD, and Kernel Change Point
(KCP) detection on small synthetic data sets.
"""

import numpy as np

from changepoint_toolkit import (
    BOCPD,
    BOCPDConfig,
    ConstantHazard,
    WithinPeriodCPD,
    ModelPrior,
    RJConfig,
    gram_rbf,
    kcp_penalized,
)
from kcp import build_kernel_prefix


def run_bocpd():
    """Run BOCPD on binary sequence with one changepoint."""
    rng = np.random.default_rng(0)
    x = np.concatenate(
        [rng.binomial(1, 0.1, 50), rng.binomial(1, 0.8, 50)]
    )
    hazard = ConstantHazard(mean_run_length=50)
    config = BOCPDConfig(alpha0=1.0, beta0=1.0, max_run_length=100)
    model = BOCPD(hazard, config)
    result = model.run(x)
    cps = np.where(result.cp_prob > 0.5)[0]
    print("BOCPD change points:", cps.tolist())


def run_within_period():
    """Run Within-Period CPD on synthetic daily data."""
    rng = np.random.default_rng(1)
    N = 24  # period length (hours in a day)
    days = 10
    x = rng.binomial(1, 0.1, N * days).astype(bool)
    # Introduce higher activity between hours 8-11
    for d in range(days):
        idx = d * N + 8
        x[idx : idx + 4] = rng.binomial(1, 0.8, 4).astype(bool)
    prior = ModelPrior(N=N, l=4)
    model = WithinPeriodCPD(prior)
    cfg = RJConfig(iters=1000, burn=200, thin=5, seed=42)
    result = model.fit(x, cfg)
    print("Within-Period CPD MAP changepoints:", result.mode_tau)


def run_kcp():
    """Run kernel change-point detection on 1D data."""
    rng = np.random.default_rng(2)
    X = np.concatenate(
        [rng.normal(0, 1, 60), rng.normal(3, 1, 60)]
    )[:, None]
    K, _ = gram_rbf(X)
    pref = build_kernel_prefix(K)
    res = kcp_penalized(pref, penalty=np.log(X.shape[0]), min_size=10, method="pelt")
    print("KCP change points:", res.change_points.tolist())


if __name__ == "__main__":
    run_bocpd()
    run_within_period()
    run_kcp()
