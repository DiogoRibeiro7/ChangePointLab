from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]


def test_mypy_checks_stable_api_surface() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    mypy = config["tool"]["mypy"]
    files = set(mypy["files"])
    assert mypy["follow_imports"] == "normal"
    assert "src/changepoint_lab/__init__.py" in files
    assert "src/changepoint_lab/core" in files
    assert "tests/typecheck" in files
    assert "src/changepoint_lab/algorithms/bayesian/bocpd" in files
    assert "src/changepoint_lab/algorithms/point_process/sliced_poisson.py" in files


def test_no_bare_type_ignores_in_checked_code() -> None:
    checked_roots = [ROOT / "src" / "changepoint_lab", ROOT / "tests" / "typecheck"]
    violations: list[str] = []
    for checked_root in checked_roots:
        for path in checked_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(r"type:\s*ignore(?!\[)", text):
                rel = path.relative_to(ROOT)
                violations.append(f"{rel}:{text[:match.start()].count(chr(10)) + 1}")
    assert violations == []
