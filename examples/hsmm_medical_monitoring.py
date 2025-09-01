"""
HSMM for Medical Monitoring
===========================

Model physiological states (e.g., resting, walking, exercising) in a
heart-rate time series using a Hidden Semi-Markov Model. We compare the
explicit-duration HSMM with a standard HMM from `hmmlearn`.
"""

import numpy as np
import matplotlib.pyplot as plt

from hsmm.gaussian_diag import estimate_by_kmeanspp, gaussian_diag_loglik
from hsmm.hsmm import HSMM, HSMMConfig, HSMMParams, PoissonDur

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    GaussianHMM = None


# ---------------------------------------------------------------------------
# Synthetic heart-rate data
# ---------------------------------------------------------------------------

def generate_hr(seed: int = 0):
    rng = np.random.default_rng(seed)
    means = [60, 90, 120]
    lengths = [200, 150, 250]
    data = np.concatenate([
        rng.normal(m, 5.0, l) for m, l in zip(means, lengths)
    ])
    cps = np.cumsum(lengths)[:-1]
    return data, cps


# ---------------------------------------------------------------------------
# HSMM fit and decode
# ---------------------------------------------------------------------------

def run_hsmm(data: np.ndarray):
    X = data[:, None]
    K = 3
    em_params = estimate_by_kmeanspp(X, K, n_init=5, max_iter=100, allow_nan=False)
    L = gaussian_diag_loglik(X, em_params)
    pi0 = np.full(K, 1.0 / K)
    A0 = np.full((K, K), 1.0 / (K - 1)); np.fill_diagonal(A0, 0.0)
    dur = ("poisson", PoissonDur(lam=np.array([60.0, 40.0, 80.0])))
    model = HSMM(HSMMConfig(K=K, Dmax=100, min_duration=10, max_em_iters=50), HSMMParams(pi=pi0, A=A0, duration=dur))
    model.fit(L)
    z, _ = model.decode_viterbi(L)
    cps = np.where(np.diff(z) != 0)[0] + 1
    return cps


# ---------------------------------------------------------------------------
# HMM comparison
# ---------------------------------------------------------------------------

def run_hmm(data: np.ndarray):
    if GaussianHMM is None:
        return []
    X = data[:, None]
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=0)
    model.fit(X)
    states = model.predict(X)
    cps = np.where(np.diff(states) != 0)[0] + 1
    return cps


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def main():
    data, true_cps = generate_hr()
    cps_hsmm = run_hsmm(data)
    cps_hmm = run_hmm(data)

    plt.figure(figsize=(10, 4))
    plt.plot(data, label="Heart rate")
    for cp in true_cps:
        plt.axvline(cp, color="k", linestyle="--", alpha=0.3)
    for cp in cps_hsmm:
        plt.axvline(cp, color="r", alpha=0.7, label="HSMM" if cp == cps_hsmm[0] else "")
    for cp in cps_hmm:
        plt.axvline(cp, color="b", linestyle="-.", alpha=0.7, label="HMM" if cp == cps_hmm[0] else "")
    plt.legend()
    plt.title("HSMM vs. HMM on heart-rate data")
    plt.xlabel("Time")
    plt.ylabel("BPM")
    plt.tight_layout()
    plt.show()

    print("True changepoints:", true_cps)
    print("HSMM detected:", cps_hsmm)
    print("HMM detected:", cps_hmm)


if __name__ == "__main__":
    main()
