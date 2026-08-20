from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RISK_REGISTER = ROOT / "docs" / "audit" / "risk_register.md"
ROADMAP = ROOT / "docs" / "audit" / "implementation_roadmap.md"


def _table_rows(markdown: str, heading: str) -> list[list[str]]:
    start = markdown.index(heading)
    next_heading = markdown.find("\n## ", start + len(heading))
    section = markdown[start:] if next_heading == -1 else markdown[start:next_heading]
    rows: list[list[str]] = []
    for line in section.splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or cells[0] == "---" or cells[0] == "ID":
            continue
        rows.append(cells)
    return rows


def _active_text() -> str:
    risk_text = RISK_REGISTER.read_text(encoding="utf-8")
    roadmap_text = ROADMAP.read_text(encoding="utf-8")
    open_risks = risk_text[
        risk_text.index("## Open Risks") : risk_text.index("## Resolved or Partially")
    ]
    current_roadmap = roadmap_text[roadmap_text.index("## Current Dependency-Aware Roadmap") :]
    return open_risks + "\n" + current_roadmap


def test_risk_ids_are_unique() -> None:
    text = RISK_REGISTER.read_text(encoding="utf-8")
    ids = re.findall(r"\|\s+(R{1,2}-\d{3})\s+\|", text)
    assert ids
    assert len(ids) == len(set(ids))


def test_risk_table_values_are_valid() -> None:
    text = RISK_REGISTER.read_text(encoding="utf-8")
    severities = {"Critical", "High", "Medium", "Low"}
    statuses = {"Resolved", "Partially resolved", "Resolved before this audit"}

    for row in _table_rows(text, "## Open Risks"):
        assert row[1] in severities
    for row in _table_rows(text, "## Resolved or Partially Resolved Findings"):
        assert row[1] in statuses


def test_active_audit_path_references_exist() -> None:
    path_prefixes = ("src/", "tests/", "docs/", ".github/", "artifacts/")
    root_files = {"pyproject.toml", "README.md", "CHANGELOG.md", "CITATION.cff", ".zenodo.json"}
    paths = []
    for token in re.findall(r"`([^`]+)`", _active_text()):
        if token.startswith(path_prefixes) or token in root_files:
            paths.append(token)
    assert paths
    missing = [path for path in paths if not (ROOT / path).exists()]
    assert missing == []


def test_new_scientific_risks_link_method_registry_entries() -> None:
    rows = {row[0]: row for row in _table_rows(RISK_REGISTER.read_text(), "## Open Risks")}
    for risk_id in ("R-021",):
        row_text = " ".join(rows[risk_id])
        assert "docs/science/method_registry.yml" in row_text or "sliced_poisson_process" in row_text
