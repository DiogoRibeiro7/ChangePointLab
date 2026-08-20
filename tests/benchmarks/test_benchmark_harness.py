from __future__ import annotations

import csv
import json

from scripts import benchmark_harness


def test_benchmark_artifact_schema() -> None:
    artifact = benchmark_harness.build_artifact(
        sizes=(32,),
        seed=20260820,
        deterministic_measurements=True,
        generated_at="2026-08-20T00:00:00Z",
    )

    assert artifact["schema_version"] == "1.0"
    assert artifact["profile"] == "smoke"
    assert len(artifact["datasets"]) == 1
    assert len(artifact["runtime"]) == 4
    assert len(artifact["accuracy"]) == 4
    assert len(artifact["approximation"]) == 1

    dataset = artifact["datasets"][0]
    assert dataset["name"] == "piecewise_gaussian_n32"
    assert dataset["hash_sha256"]
    assert dataset["true_changepoints"] == (16,)

    for record in artifact["runtime"]:
        assert record["elapsed_ns"] == 0
        assert record["peak_memory_bytes"] == 0
        assert record["dataset"] == dataset["name"]
        assert isinstance(record["config"], dict)


def test_benchmark_regeneration_is_deterministic() -> None:
    first = benchmark_harness.build_artifact(
        sizes=(32, 64),
        seed=20260820,
        deterministic_measurements=True,
        generated_at="2026-08-20T00:00:00Z",
    )
    second = benchmark_harness.build_artifact(
        sizes=(32, 64),
        seed=20260820,
        deterministic_measurements=True,
        generated_at="2026-08-20T00:00:00Z",
    )

    assert first == second


def test_benchmark_cli_writes_json_csv_and_markdown(tmp_path) -> None:
    output_dir = tmp_path / "artifacts"
    report = tmp_path / "report.md"

    code = benchmark_harness.main(
        [
            "--profile",
            "smoke",
            "--output-dir",
            str(output_dir),
            "--report",
            str(report),
            "--deterministic-measurements",
        ]
    )

    assert code == 0
    payload = json.loads((output_dir / "smoke_benchmark.json").read_text(encoding="utf-8"))
    with (output_dir / "smoke_benchmark.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    text = report.read_text(encoding="utf-8")

    assert payload["schema_version"] == "1.0"
    assert rows
    assert "# Benchmark Smoke Artifact" in text
    assert "Real local timing and memory" in text


def test_committed_smoke_artifacts_are_schema_valid() -> None:
    artifact_path = benchmark_harness.DEFAULT_OUTPUT_DIR / "smoke_benchmark.json"
    csv_path = benchmark_harness.DEFAULT_OUTPUT_DIR / "smoke_benchmark.csv"
    report_path = benchmark_harness.DEFAULT_REPORT

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert artifact["schema_version"] == "1.0"
    assert artifact["profile"] == "smoke"
    assert artifact["environment"]["commit"]
    assert artifact["datasets"]
    assert artifact["runtime"]
    assert artifact["accuracy"]
    assert artifact["approximation"]
    assert rows
    assert report_path.exists()
