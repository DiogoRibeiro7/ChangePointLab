from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE_DOC = ROOT / "docs" / "architecture" / "index.md"


def _current_architecture_section() -> str:
    text = ARCHITECTURE_DOC.read_text(encoding="utf-8")
    start = "<!-- architecture-current-start -->"
    end = "<!-- architecture-current-end -->"
    return text[text.index(start) + len(start) : text.index(end)]


def test_current_architecture_paths_exist() -> None:
    section = _current_architecture_section()
    paths = [
        token
        for token in re.findall(r"`([^`]+)`", section)
        if token.startswith(("src/", "tests/", "docs/"))
    ]

    assert paths
    missing = [path for path in paths if not (ROOT / path).exists()]
    assert missing == []


def test_architecture_separates_current_and_target_structure() -> None:
    text = ARCHITECTURE_DOC.read_text(encoding="utf-8")

    assert "## Current Source Layout" in text
    assert "## Target Architecture" in text
    assert text.index("## Current Source Layout") < text.index("## Target Architecture")
    assert "`src/changepoint_lab/algorithms/bayesian/bocpd/` is a package" in text
    assert "should not be described as present until implemented" in text
