from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_repository_community_files_exist() -> None:
    expected = [
        ".github/CODEOWNERS",
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/ISSUE_TEMPLATE/scientific_validation.yml",
        ".github/ISSUE_TEMPLATE/documentation.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
        ".github/workflows/codeql.yml",
        ".github/workflows/dependency-review.yml",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "SUPPORT.md",
    ]

    missing = [path for path in expected if not (ROOT / path).is_file()]
    assert missing == []


def test_issue_forms_collect_required_triage_fields() -> None:
    forms = {
        "bug_report.yml": ["name: Bug report", "id: reproduce", "id: version"],
        "feature_request.yml": ["name: Feature request", "id: problem", "id: proposal"],
        "scientific_validation.yml": [
            "name: Scientific validation",
            "id: evidence",
            "id: reproduction",
        ],
        "documentation.yml": ["name: Documentation issue", "id: location", "id: issue"],
    }

    root = ROOT / ".github" / "ISSUE_TEMPLATE"
    for filename, markers in forms.items():
        text = (root / filename).read_text(encoding="utf-8")
        for marker in markers:
            assert marker in text


def test_security_and_pr_templates_point_to_project_checks() -> None:
    pr_template = (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")
    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")

    assert "poetry run ruff check ." in pr_template
    assert "poetry run mypy" in pr_template
    assert "private vulnerability reporting" in security
