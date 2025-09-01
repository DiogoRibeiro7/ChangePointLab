"""Reusable helpers for multi-method changepoint comparisons."""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def runtime(fn: Callable[[], Sequence[int]]) -> Tuple[List[int], float]:
    """Execute ``fn`` returning its changepoints and runtime."""
    start = time.perf_counter()
    cps = list(fn())
    return cps, time.perf_counter() - start


def f1_score(pred: Sequence[int], true: Sequence[int], tol: int = 5) -> Dict[str, float]:
    """Precision/recall/F1 with tolerance window ``tol``."""
    true = list(true)
    matched = np.zeros(len(true), dtype=bool)
    tp = 0
    for p in pred:
        if not true:
            break
        idx = np.argmin(np.abs(np.array(true) - p))
        if abs(true[idx] - p) <= tol and not matched[idx]:
            matched[idx] = True
            tp += 1
    fp = len(pred) - tp
    fn = len(true) - tp
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def plot_series(ax: plt.Axes, series: Sequence[float], pred_cps: Sequence[int], true_cps: Sequence[int], label: str) -> None:
    """Plot ``series`` with predicted and true changepoints."""
    ax.plot(series, label=label)
    for cp in pred_cps:
        ax.axvline(cp, color="red", linestyle="--", alpha=0.5)
    for cp in true_cps:
        ax.axvline(cp, color="green", linestyle=":", alpha=0.5)
    ax.legend()


def compare_detectors(
    name: str,
    detectors: Sequence[Tuple[str, Callable[[], Sequence[int]], Sequence[float]]],
    true_cps: Sequence[int],
    *,
    tol: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Run ``detectors`` on their associated series and create comparison plots.

    Parameters
    ----------
    name : str
        Scenario name used for plot titles and file names.
    detectors : sequence of ``(label, fn, series)``
        ``fn`` returns changepoints for ``series``.
    true_cps : sequence of int
        Ground truth changepoints.
    tol : int, optional
        Tolerance window for F1 score.

    Returns
    -------
    dict
        Mapping from detector label to metric dictionary including precision,
        recall, F1 and runtime.
    """
    cols = 3
    rows = math.ceil(len(detectors) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    metrics: Dict[str, Dict[str, float]] = {}
    for ax, (label, fn, series) in zip(axes, detectors):
        cps, t = runtime(fn)
        stats = f1_score(cps, true_cps, tol=tol)
        metrics[label] = {**stats, "runtime": t}
        plot_series(ax, series, cps, true_cps, f"{label}\nF1={stats['f1']:.2f}, {t:.2f}s")

    for ax in axes[len(detectors) :]:
        ax.axis("off")

    fig.suptitle(f"Scenario: {name}; true CPs {list(true_cps)}")
    plt.tight_layout()
    plt.savefig(f"{name}_comparison.png")
    plt.close(fig)
    return metrics


def print_summary(name: str, metrics: Dict[str, Dict[str, float]], discussion: str | None = None) -> None:
    """Pretty-print metric table and optional discussion for a scenario."""
    print(f"\nScenario: {name}")
    for label, stats in metrics.items():
        print(
            f"  {label:15s} F1={stats['f1']:.2f} Precision={stats['precision']:.2f} "
            f"Recall={stats['recall']:.2f} Runtime={stats['runtime']:.3f}s"
        )
    if metrics:
        best = max(metrics.items(), key=lambda kv: kv[1]["f1"])[0]
        if discussion:
            print(f"  -> Best: {best}. {discussion}")
        else:
            print(f"  -> Best: {best}")
