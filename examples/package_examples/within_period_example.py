"""
Within-Period Detection for Energy Consumption
=============================================

Analyze daily electricity usage with within-period changepoint detection
based on an RJMCMC sampler. We compare results with BOCPD using a
periodic hazard.
"""

import numpy as np
import matplotlib.pyplot as plt

from changepoint_lab.algorithms.bayesian.within_period import (
    ModelPrior,
    RJConfig,
    WithinPeriodCPD,
)
from changepoint_lab.algorithms.bayesian.bocpd import (
    BOCPD,
    BoostedBoundaryHazard,
)


# ---------------------------------------------------------------------------
# Synthetic daily energy data
# ---------------------------------------------------------------------------

def generate_energy(seed: int = 0, days: int = 30, bins_per_day: int = 24):
    rng = np.random.default_rng(seed)
    N = bins_per_day
    baseline = np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False)) * 0.3 + 0.5
    shift = np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False) + np.pi / 4) * 0.3 + 0.5
    day_pattern = np.vstack([baseline, shift])
    pattern_idx = np.repeat([0, 0, 0, 1, 1, 1], days // 6)
    data = np.concatenate([rng.poisson(day_pattern[i]) for i in pattern_idx])
    data = (data > np.median(data)).astype(int)
    cps = np.cumsum([N] * len(pattern_idx))[:-1]
    return data, cps, N


# ---------------------------------------------------------------------------
# Within-period CPD
# ---------------------------------------------------------------------------

def run_within_period(data: np.ndarray, N: int):
    prior = ModelPrior(N=N, l=2, gamma=1.0, pois_lambda=1.0)
    model = WithinPeriodCPD(prior)
    cfg = RJConfig(iters=2000, burn=1000, thin=10, seed=0)
    res = model.fit(data, cfg)
    return list(res.mode_tau)


# ---------------------------------------------------------------------------
# BOCPD comparison with periodic hazard
# ---------------------------------------------------------------------------

def run_bocpd(data: np.ndarray, N: int):
    hazard = BoostedBoundaryHazard(mean_run_length=50, boundaries=np.arange(0, len(data), N))
    model = BOCPD(hazard=hazard)
    cps = []
    for t, x in enumerate(data):
        if model.update(x) > 0.5:
            cps.append(t)
    return cps


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def main():
    data, true_cps, N = generate_energy()
    cps_wp = run_within_period(data, N)
    cps_bocpd = run_bocpd(data, N)

    plt.figure(figsize=(10, 4))
    plt.step(range(len(data)), data, where="post", label="Usage")
    for cp in true_cps:
        plt.axvline(cp, color="k", linestyle="--", alpha=0.3)
    for cp in cps_wp:
        plt.axvline(cp, color="r", alpha=0.7, label="Within-period" if cp == cps_wp[0] else "")
    for cp in cps_bocpd:
        plt.axvline(cp, color="b", linestyle="-.", alpha=0.7, label="BOCPD" if cp == cps_bocpd[0] else "")
    plt.legend()
    plt.title("Within-period vs. BOCPD for energy usage")
    plt.xlabel("Time")
    plt.ylabel("Binary usage")
    plt.tight_layout()
    plt.show()

    print("True changepoints:", true_cps)
    print("Within-period detected:", cps_wp)
    print("BOCPD detected:", cps_bocpd)


if __name__ == "__main__":
    main()
