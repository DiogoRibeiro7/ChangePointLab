"""Hybrid online/offline changepoint detection.

BOCPD provides low-latency online alerts while PELT periodically verifies the
stream in batch mode.  The offline analysis can feed back into BOCPD by updating
its hazard rate, reducing false positives and drift.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from bocpd import BOCPD, ConstantHazard
from pelt import pelt, NormalMeanVarUnknown
from examples.comparison_helpers import f1_score, plot_series


def generate_stream(seed: int = 4):
    rng = np.random.default_rng(seed)
    segs = [rng.normal(0, 1, 100), rng.normal(3, 1, 120), rng.normal(-2, 1, 100)]
    data = np.concatenate(segs)
    truth = np.cumsum([len(s) for s in segs])[:-1]
    return data, truth


def main():
    data, truth = generate_stream()
    hazard = ConstantHazard(mean_run_length=80)
    model = BOCPD(hazard=hazard, alpha=1.0, beta=1.0)

    online_cps = []
    posteriors = []
    for i, x in enumerate(data, start=1):
        cp_prob = model.update(x)
        posteriors.append(cp_prob)
        if cp_prob > 0.5:
            online_cps.append(i)
        # every 100 samples run PELT offline
        if i % 100 == 0:
            offline_cps = pelt(data[:i], NormalMeanVarUnknown(), penalty=10.0)
            if offline_cps:
                # adjust hazard to match observed segment length
                last_cp = offline_cps[-1]
                seg_len = i - last_cp
                hazard.mean_run_length = 0.8 * seg_len + 0.2 * hazard.mean_run_length

    metrics = {
        "BOCPD online": f1_score(online_cps, truth),
        "PELT offline": f1_score(pelt(data, NormalMeanVarUnknown(), penalty=10.0), truth),
    }
    print(metrics)

    fig, ax = plt.subplots(figsize=(8, 3))
    plot_series(ax, data, online_cps, truth, label="Online BOCPD")
    ax.set_title("Hybrid online/offline detection")
    plt.show()

    print("Hybrid detection yields quick alerts with BOCPD while offline PELT "
          "periodically recalibrates the hazard to maintain accuracy.")


if __name__ == "__main__":
    main()
