"""Two-stage changepoint detection combining PELT and BOCPD.

This example illustrates a workflow where a fast offline method (PELT)
provides coarse segmentation which is then refined by an online method
(BOCPD).  The combined approach can yield more accurate changepoint
locations than either technique alone.

Steps
-----
1. Generate a noisy signal with two mean shifts.
2. Run PELT with a strong penalty to find coarse changepoints.
3. Run BOCPD within each PELT segment for fine-grained detection.
4. Evaluate F1 scores and runtime against ground truth and single methods.
5. Plot the results for visual comparison.
"""
from __future__ import annotations

import time
import numpy as np
import matplotlib.pyplot as plt

from pelt import pelt, NormalMeanVarUnknown
from changepoint_lab.algorithms.bayesian.bocpd import BOCPD, ConstantHazard
from examples.comparison_helpers import f1_score, runtime, plot_series


def generate_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    segments = [rng.normal(0, 1, 100), rng.normal(5, 1, 80), rng.normal(2, 1, 120)]
    data = np.concatenate(segments)
    cps = np.cumsum([len(s) for s in segments])[:-1]
    return data, cps


def coarse_pelt(data: np.ndarray):
    cost = NormalMeanVarUnknown()
    return pelt(data, cost, penalty=10.0)


def refine_bocpd(data: np.ndarray, coarse_cps: list[int]):
    hazard = ConstantHazard(mean_run_length=50)
    model = BOCPD(hazard=hazard, alpha=1.0, beta=1.0)
    refined: list[int] = []
    last = 0
    for cp in coarse_cps + [len(data)]:
        segment = data[last:cp]
        if len(segment) > 0:
            sub_cps = model.fit_predict(segment)
            refined.extend(last + np.array(sub_cps, dtype=int))
        last = cp
    return sorted(set(refined))


def main():
    data, truth = generate_data()

    # PELT coarse segmentation
    cps_pelt, t_pelt = runtime(lambda: coarse_pelt(data))

    # BOCPD over full series
    hazard = ConstantHazard(mean_run_length=50)
    bocpd_model = BOCPD(hazard=hazard, alpha=1.0, beta=1.0)
    cps_bocpd, t_bocpd = runtime(lambda: bocpd_model.fit_predict(data))

    # Two-stage: refine PELT segments with BOCPD
    start = time.perf_counter()
    cps_refined = refine_bocpd(data, cps_pelt)
    t_refined = time.perf_counter() - start

    # Evaluate
    metrics = {
        "PELT": {**f1_score(cps_pelt, truth), "runtime": t_pelt},
        "BOCPD": {**f1_score(cps_bocpd, truth), "runtime": t_bocpd},
        "Two-stage": {**f1_score(cps_refined, truth), "runtime": t_refined},
    }
    for name, m in metrics.items():
        print(f"{name:9s} F1={m['f1']:.2f} Precision={m['precision']:.2f} "
              f"Recall={m['recall']:.2f} Runtime={m['runtime']:.3f}s")

    # Visualise
    fig, ax = plt.subplots(figsize=(8, 3))
    plot_series(ax, data, cps_refined, truth, label="Two-stage")
    ax.set_title("Two-stage detection vs truth")
    plt.show()

    print("Two-stage detection is helpful when we trust PELT to locate rough "
          "regions of change but need BOCPD's online precision within each "
          "segment.")


if __name__ == "__main__":
    main()
