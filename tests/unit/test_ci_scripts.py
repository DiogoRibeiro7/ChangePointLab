from __future__ import annotations

import pytest

from scripts.validate_coverage_policy import validate_coverage_report


def _file(statements: int, covered: int) -> dict[str, dict[str, int]]:
    return {"summary": {"num_statements": statements, "covered_lines": covered}}


def _coverage_report(total: float) -> dict[str, object]:
    return {
        "totals": {"percent_covered": total},
        "files": {
            "src/changepoint_lab/core/datatypes.py": _file(100, 95),
            "src/changepoint_lab/algorithms/optimization/pelt.py": _file(100, 92),
            "src/changepoint_lab/algorithms/nonparametric/edivisive_core.py": _file(100, 91),
            "src/changepoint_lab/algorithms/point_process/sliced_poisson.py": _file(100, 86),
            "src/changepoint_lab/algorithms/bayesian/bocpd/core.py": _file(100, 82),
        },
    }


def test_coverage_policy_accepts_current_floor_shape() -> None:
    validate_coverage_report(_coverage_report(66.0))


def test_coverage_policy_fails_on_total_regression() -> None:
    with pytest.raises(RuntimeError, match="overall coverage"):
        validate_coverage_report(_coverage_report(64.9))


def test_coverage_policy_fails_on_core_regression() -> None:
    report = _coverage_report(66.0)
    files = report["files"]
    assert isinstance(files, dict)
    files["src/changepoint_lab/core/datatypes.py"] = _file(100, 89)

    with pytest.raises(RuntimeError, match="core contracts"):
        validate_coverage_report(report)
