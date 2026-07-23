from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from changepoint_lab.algorithms.bayesian.within_period import (
    mysense_sensor_example,
    paper_replication_scenarios,
    simulate_periodic_bernoulli,
    write_reproduction_artifacts,
)
from changepoint_lab.algorithms.bayesian.within_period.within_period_cpd import _is_valid_tau


ROOT = Path(__file__).resolve().parents[2]


def test_paper_replication_scenarios_cover_required_cases() -> None:
    scenarios = paper_replication_scenarios()
    names = {scenario.name for scenario in scenarios}

    assert {
        "paper_monte_carlo_n24",
        "no_change",
        "one_activity_window",
        "multiple_activity_windows",
        "weak_signal",
        "boundary_crossing_sleep",
    } <= names
    for scenario in scenarios:
        assert scenario.source_scope == "paper_consistent"
        assert scenario.period % scenario.min_segment_length == 0
        assert _is_valid_tau(
            scenario.boundaries,
            scenario.period,
            scenario.min_segment_length,
        )
        simulated = simulate_periodic_bernoulli(scenario, seed=123)
        assert simulated.dtype == np.bool_
        assert simulated.shape == (scenario.period * scenario.days,)


def test_mysense_example_has_named_sensor_streams_and_aggregate() -> None:
    example = mysense_sensor_example(days=3, seed=5)

    assert set(example.sensors) == {"chair", "doors", "kettle", "tap", "toilet"}
    assert example.source_scope == "mysense_extension"
    for values in example.sensors.values():
        assert values.shape == (example.period * example.days,)
    assert np.array_equal(
        example.any_activity,
        np.any(np.vstack(list(example.sensors.values())), axis=0),
    )


def test_reproduction_artifact_writer_separates_paper_and_mysense_outputs(tmp_path: Path) -> None:
    artifacts = write_reproduction_artifacts(tmp_path, profile="ci")

    expected = {
        "summary",
        "paper_scenario_summary",
        "prior_sensitivity",
        "mysense_sensor_rates",
        "paper_changepoint_mass",
    }
    assert set(artifacts) == expected
    for path in artifacts.values():
        assert path.exists()

    summary = json.loads(artifacts["summary"].read_text(encoding="utf-8"))
    assert summary["profile"]["name"] == "ci"
    assert len(summary["paper_consistent"]) >= 6
    assert summary["mysense_extension"]["summary"]["source_scope"] == "mysense_extension"
    assert summary["discrepancies"]


def test_reproduction_script_runs_from_one_command(tmp_path: Path) -> None:
    output = tmp_path / "artifacts"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_within_period_reproduction.py",
            "--profile",
            "ci",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert Path(payload["summary"]).exists()


def test_within_period_reproduction_notebook_executes() -> None:
    notebook = json.loads(
        (ROOT / "docs" / "notebooks" / "within_period_reproduction.ipynb").read_text(
            encoding="utf-8"
        )
    )
    namespace: dict[str, object] = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)

    artifacts = namespace["artifacts"]
    assert artifacts["summary"].exists()
