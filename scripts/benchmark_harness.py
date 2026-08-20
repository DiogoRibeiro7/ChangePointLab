"""Generate reproducible benchmark artifacts for ChangePointLab."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from changepoint_lab import EDivisive, KernelCPD, PELT
from changepoint_lab.algorithms.kernel import RFFConfig
from changepoint_lab.algorithms.optimization.pelt import NormalMeanKnownVar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "comparisons" / "artifacts"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "comparisons" / "benchmark_smoke.md"


@dataclass(frozen=True)
class DatasetSpec:
    """Controlled synthetic dataset specification."""

    name: str
    kind: str
    size: int
    seed: int
    true_changepoints: tuple[int, ...]
    hash_sha256: str


@dataclass(frozen=True)
class RunRecord:
    """Runtime and memory observation for one detector on one dataset."""

    detector: str
    dataset: str
    size: int
    seed: int
    elapsed_ns: int
    peak_memory_bytes: int
    predicted_changepoints: tuple[int, ...]
    score: float
    config: dict[str, object]


@dataclass(frozen=True)
class AccuracyRecord:
    """Boundary accuracy metric separated from implementation timing."""

    detector: str
    dataset: str
    tolerance: int
    precision: float
    recall: float
    f1: float
    mean_abs_error: float | None
    true_changepoints: tuple[int, ...]
    predicted_changepoints: tuple[int, ...]


@dataclass(frozen=True)
class ApproximationRecord:
    """Approximate-vs-exact comparison separated from runtime."""

    approximate_detector: str
    baseline_detector: str
    dataset: str
    boundary_f1: float
    score_abs_error: float
    score_relative_error: float


def _git_value(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(  # noqa: S603
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_metadata(*, generated_at: str, deterministic: bool) -> dict[str, object]:
    """Return reproducibility metadata for a benchmark artifact."""
    commit = _git_value(["rev-parse", "HEAD"])
    status = _git_value(["status", "--short"])
    return {
        "generated_at": generated_at,
        "commit": commit,
        "working_tree_dirty": bool(status),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "deterministic_measurements": deterministic,
    }


def piecewise_gaussian(size: int, *, seed: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Generate a deterministic mean-shift sequence."""
    if size < 12:
        raise ValueError("size must be at least 12.")
    rng = np.random.default_rng(seed)
    cp = size // 2
    x = np.concatenate(
        [
            rng.normal(0.0, 0.15, size=cp),
            rng.normal(3.0, 0.15, size=size - cp),
        ]
    )
    return x.astype(float), (cp,)


def dataset_hash(data: np.ndarray) -> str:
    """Return a stable hash for a benchmark dataset."""
    payload = np.ascontiguousarray(data, dtype=np.float64)
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _as_tuple(values: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in np.asarray(values, dtype=int).tolist())


def _pelt_detector(data: np.ndarray) -> tuple[tuple[int, ...], float, dict[str, object]]:
    model = PELT(cost_fn=NormalMeanKnownVar(sigma2=1.0), penalty=6.0, min_seg_len=4)
    result = model.fit_predict(data)
    return _as_tuple(result.indices), float(result.score), {
        "cost": "NormalMeanKnownVar",
        "sigma2": 1.0,
        "penalty": 6.0,
        "min_seg_len": 4,
    }


def _kernel_exact_detector(data: np.ndarray) -> tuple[tuple[int, ...], float, dict[str, object]]:
    x = data.reshape(-1, 1)
    model = KernelCPD(penalty=0.5, min_size=4, method="op", bandwidth=1.0)
    result = model.fit_predict(x)
    return _as_tuple(result.indices), float(result.score), {
        "kernel": "rbf",
        "approximation": "exact",
        "penalty": 0.5,
        "min_size": 4,
        "method": "op",
        "bandwidth": 1.0,
    }


def _kernel_rff_detector(data: np.ndarray) -> tuple[tuple[int, ...], float, dict[str, object]]:
    x = data.reshape(-1, 1)
    cfg = RFFConfig(n_features=32, seed=123, subsample_for_bandwidth=64)
    model = KernelCPD(
        penalty=0.5,
        min_size=4,
        method="op",
        bandwidth=1.0,
        approximation="rff",
        rff_config=cfg,
    )
    result = model.fit_predict(x)
    return _as_tuple(result.indices), float(result.score), {
        "kernel": "rbf",
        "approximation": "rff",
        "rff_n_features": cfg.n_features,
        "rff_seed": cfg.seed,
        "penalty": 0.5,
        "min_size": 4,
        "method": "op",
        "bandwidth": 1.0,
    }


def _edivisive_detector(data: np.ndarray) -> tuple[tuple[int, ...], float, dict[str, object]]:
    model = EDivisive(min_size=4, R=19, seed=123, significance=0.2, max_cps=1)
    result = model.fit_predict(data)
    score = 0.0 if result.score is None else float(result.score)
    return _as_tuple(result.indices), score, {
        "min_size": 4,
        "R": 19,
        "seed": 123,
        "significance": 0.2,
        "max_cps": 1,
    }


DETECTORS: dict[str, Callable[[np.ndarray], tuple[tuple[int, ...], float, dict[str, object]]]] = {
    "pelt_gaussian": _pelt_detector,
    "kernel_exact": _kernel_exact_detector,
    "kernel_rff": _kernel_rff_detector,
    "edivisive": _edivisive_detector,
}


def boundary_metrics(
    truth: Sequence[int],
    predicted: Sequence[int],
    *,
    tolerance: int,
) -> tuple[float, float, float, float | None]:
    """Compute one-to-one boundary precision, recall, F1, and mean absolute error."""
    unmatched = [int(v) for v in truth]
    errors: list[int] = []
    matches = 0
    for pred in predicted:
        if not unmatched:
            break
        distances = [abs(int(pred) - true) for true in unmatched]
        best_idx = int(np.argmin(distances))
        best_error = distances[best_idx]
        if best_error <= tolerance:
            matches += 1
            errors.append(best_error)
            unmatched.pop(best_idx)
    precision = matches / len(predicted) if predicted else (1.0 if not truth else 0.0)
    recall = matches / len(truth) if truth else (1.0 if not predicted else 0.0)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    mean_abs_error = float(np.mean(errors)) if errors else None
    return precision, recall, f1, mean_abs_error


def run_once(
    detector_name: str,
    data: np.ndarray,
    *,
    deterministic_measurements: bool,
) -> tuple[tuple[int, ...], float, int, int, dict[str, object]]:
    """Run one detector and return predictions plus timing and memory."""
    detector = DETECTORS[detector_name]
    if deterministic_measurements:
        predicted, score, config = detector(data)
        return predicted, score, 0, 0, config

    tracemalloc.start()
    started = time.perf_counter_ns()
    try:
        predicted, score, config = detector(data)
        elapsed_ns = time.perf_counter_ns() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return predicted, score, int(elapsed_ns), int(peak), config


def build_artifact(
    *,
    sizes: Sequence[int],
    seed: int,
    deterministic_measurements: bool,
    generated_at: str,
) -> dict[str, object]:
    """Build a benchmark artifact dictionary from controlled synthetic datasets."""
    datasets: list[DatasetSpec] = []
    runs: list[RunRecord] = []
    accuracy: list[AccuracyRecord] = []
    approximations: list[ApproximationRecord] = []

    for size in sizes:
        data, truth = piecewise_gaussian(size, seed=seed + size)
        dataset = DatasetSpec(
            name=f"piecewise_gaussian_n{size}",
            kind="piecewise_gaussian",
            size=size,
            seed=seed + size,
            true_changepoints=truth,
            hash_sha256=dataset_hash(data),
        )
        datasets.append(dataset)
        by_detector: dict[str, RunRecord] = {}

        for detector_name in DETECTORS:
            predicted, score, elapsed_ns, peak_memory_bytes, config = run_once(
                detector_name,
                data,
                deterministic_measurements=deterministic_measurements,
            )
            record = RunRecord(
                detector=detector_name,
                dataset=dataset.name,
                size=size,
                seed=dataset.seed,
                elapsed_ns=elapsed_ns,
                peak_memory_bytes=peak_memory_bytes,
                predicted_changepoints=predicted,
                score=score,
                config=config,
            )
            runs.append(record)
            by_detector[detector_name] = record
            precision, recall, f1, mae = boundary_metrics(truth, predicted, tolerance=2)
            accuracy.append(
                AccuracyRecord(
                    detector=detector_name,
                    dataset=dataset.name,
                    tolerance=2,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    mean_abs_error=mae,
                    true_changepoints=truth,
                    predicted_changepoints=predicted,
                )
            )

        exact = by_detector["kernel_exact"]
        approx = by_detector["kernel_rff"]
        _, _, boundary_f1, _ = boundary_metrics(
            exact.predicted_changepoints,
            approx.predicted_changepoints,
            tolerance=2,
        )
        score_abs_error = abs(approx.score - exact.score)
        approximations.append(
            ApproximationRecord(
                approximate_detector="kernel_rff",
                baseline_detector="kernel_exact",
                dataset=dataset.name,
                boundary_f1=boundary_f1,
                score_abs_error=score_abs_error,
                score_relative_error=score_abs_error / max(abs(exact.score), 1e-12),
            )
        )

    return {
        "schema_version": "1.0",
        "profile": "smoke" if max(sizes) <= 96 else "full",
        "environment": environment_metadata(
            generated_at=generated_at,
            deterministic=deterministic_measurements,
        ),
        "datasets": [asdict(item) for item in datasets],
        "runtime": [asdict(item) for item in runs],
        "accuracy": [asdict(item) for item in accuracy],
        "approximation": [asdict(item) for item in approximations],
    }


def write_json(path: Path, artifact: dict[str, object]) -> None:
    """Persist a benchmark artifact as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, artifact: dict[str, object]) -> None:
    """Persist benchmark metrics as a flat CSV table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["section", "dataset", "detector", "metric", "value", "unit"]
    rows: list[dict[str, object]] = []
    for record in artifact["runtime"]:  # type: ignore[index]
        rows.extend(
            [
                {
                    "section": "runtime",
                    "dataset": record["dataset"],
                    "detector": record["detector"],
                    "metric": "elapsed",
                    "value": record["elapsed_ns"],
                    "unit": "ns",
                },
                {
                    "section": "runtime",
                    "dataset": record["dataset"],
                    "detector": record["detector"],
                    "metric": "peak_memory",
                    "value": record["peak_memory_bytes"],
                    "unit": "bytes",
                },
            ]
        )
    for record in artifact["accuracy"]:  # type: ignore[index]
        rows.append(
            {
                "section": "accuracy",
                "dataset": record["dataset"],
                "detector": record["detector"],
                "metric": "boundary_f1",
                "value": record["f1"],
                "unit": "ratio",
            }
        )
    for record in artifact["approximation"]:  # type: ignore[index]
        rows.append(
            {
                "section": "approximation",
                "dataset": record["dataset"],
                "detector": record["approximate_detector"],
                "metric": "score_relative_error",
                "value": record["score_relative_error"],
                "unit": "ratio",
            }
        )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, artifact: dict[str, object]) -> None:
    """Generate a Markdown report from a persisted artifact."""
    env = artifact["environment"]  # type: ignore[index]
    lines = [
        "# Benchmark Smoke Artifact",
        "",
        "This report is generated from committed JSON and CSV artifacts. It is a",
        "small harness smoke check, not a performance ranking.",
        "",
        "## Environment",
        "",
        f"- Commit: `{env['commit']}`",
        f"- Dirty tree while generated: `{env['working_tree_dirty']}`",
        f"- Python: `{env['python']}`",
        f"- NumPy: `{env['numpy']}`",
        f"- Platform: `{env['platform']}`",
        f"- Deterministic measurements: `{env['deterministic_measurements']}`",
        "",
        "## Runtime And Memory",
        "",
        "| Dataset | Detector | Elapsed ns | Peak memory bytes |",
        "| --- | --- | ---: | ---: |",
    ]
    for record in artifact["runtime"]:  # type: ignore[index]
        lines.append(
            f"| `{record['dataset']}` | `{record['detector']}` | "
            f"{record['elapsed_ns']} | {record['peak_memory_bytes']} |"
        )

    lines.extend(
        [
            "",
            "## Boundary Accuracy",
            "",
            "| Dataset | Detector | F1 | Predicted | Truth |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for record in artifact["accuracy"]:  # type: ignore[index]
        lines.append(
            f"| `{record['dataset']}` | `{record['detector']}` | {record['f1']:.3f} | "
            f"`{record['predicted_changepoints']}` | `{record['true_changepoints']}` |"
        )

    lines.extend(
        [
            "",
            "## Approximation Error",
            "",
            "| Dataset | Approximate | Baseline | Boundary F1 | Score relative error |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for record in artifact["approximation"]:  # type: ignore[index]
        lines.append(
            f"| `{record['dataset']}` | `{record['approximate_detector']}` | "
            f"`{record['baseline_detector']}` | {record['boundary_f1']:.3f} | "
            f"{record['score_relative_error']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "Smoke artifact:",
            "",
            "```bash",
            "python scripts/benchmark_harness.py --profile smoke --deterministic-measurements",
            "```",
            "",
            "Real local timing and memory:",
            "",
            "```bash",
            "python scripts/benchmark_harness.py --profile full",
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--generated-at",
        default="2026-08-20T00:00:00Z",
        help="Timestamp stored in generated artifacts.",
    )
    parser.add_argument(
        "--deterministic-measurements",
        action="store_true",
        help="Write zero timing and memory values for deterministic smoke artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark harness."""
    args = parse_args(argv)
    sizes = (32, 64) if args.profile == "smoke" else (64, 128, 256)
    artifact = build_artifact(
        sizes=sizes,
        seed=args.seed,
        deterministic_measurements=bool(args.deterministic_measurements),
        generated_at=str(args.generated_at),
    )
    json_path = args.output_dir / f"{args.profile}_benchmark.json"
    csv_path = args.output_dir / f"{args.profile}_benchmark.csv"
    write_json(json_path, artifact)
    write_csv(csv_path, artifact)
    write_markdown(args.report, artifact)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
