#!/usr/bin/env python3
"""Validate repository coverage thresholds from coverage.py JSON output."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OVERALL_FLOOR = 65.0


@dataclass(frozen=True)
class CoverageFloor:
    """Minimum line coverage for a path prefix."""

    label: str
    prefix: str
    minimum: float


GROUP_FLOORS = (
    CoverageFloor("core contracts", "src/changepoint_lab/core/", 90.0),
    CoverageFloor("optimization methods", "src/changepoint_lab/algorithms/optimization/", 90.0),
    CoverageFloor("nonparametric methods", "src/changepoint_lab/algorithms/nonparametric/", 90.0),
    CoverageFloor("point-process methods", "src/changepoint_lab/algorithms/point_process/", 85.0),
    CoverageFloor("BOCPD methods", "src/changepoint_lab/algorithms/bayesian/bocpd/", 75.0),
)


def _normalized(path: str) -> str:
    return path.replace("\\", "/")


def _percent(covered: int, statements: int) -> float:
    if statements == 0:
        return 100.0
    return covered * 100.0 / statements


def _group_coverage(files: dict[str, Any], prefix: str) -> tuple[float, int, int]:
    statements = 0
    covered = 0
    for path, payload in files.items():
        if not _normalized(path).startswith(prefix):
            continue
        summary = payload["summary"]
        statements += int(summary["num_statements"])
        covered += int(summary["covered_lines"])

    if statements == 0:
        raise RuntimeError(f"coverage group has no measured files: {prefix}")
    return _percent(covered, statements), covered, statements


def validate_coverage_report(report: dict[str, Any]) -> None:
    """Validate overall and selected package-area coverage thresholds."""

    failures: list[str] = []
    total = float(report["totals"]["percent_covered"])
    if total < OVERALL_FLOOR:
        failures.append(f"overall coverage {total:.2f}% is below {OVERALL_FLOOR:.2f}%")

    files = report["files"]
    for floor in GROUP_FLOORS:
        value, covered, statements = _group_coverage(files, floor.prefix)
        if value < floor.minimum:
            failures.append(
                f"{floor.label} coverage {value:.2f}% ({covered}/{statements}) "
                f"is below {floor.minimum:.2f}%"
            )

    if failures:
        raise RuntimeError("\n".join(failures))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coverage_json", type=Path, nargs="?", default=Path("coverage.json"))
    args = parser.parse_args(argv)

    with args.coverage_json.open(encoding="utf-8") as f:
        report = json.load(f)
    validate_coverage_report(report)
    print("coverage policy validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
