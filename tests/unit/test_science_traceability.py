from __future__ import annotations

import json
import re
from pathlib import Path

import changepoint_lab as cpl


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "docs" / "science" / "method_registry.yml"


def _load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_stable_package_algorithms_have_registry_entries() -> None:
    registry = _load_registry()
    symbol_to_method = {
        symbol: method["id"]
        for method in registry["methods"]
        for symbol in method["stable_package_symbols"]
    }
    expected_symbols = {
        "PELT",
        "BOCPD",
        "BOCPDConfig",
        "BOCPDResult",
        "Hazard",
        "ConstantHazard",
        "BoostedBoundaryHazard",
        "ScheduledHazard",
        "WithinPeriodCPD",
        "edivisive",
        "EDivisive",
        "HSMM",
        "HSMMConfig",
        "HSMMParams",
        "PoissonDur",
        "SDHMM",
        "SDHMMConfig",
        "SDHMMResult",
        "SDHMMMixVI",
        "SDHMMMixVIConfig",
        "SDHMMMixVIResult",
        "KernelCPD",
        "gram_rbf",
        "kcp_penalized",
        "kcp_select_bic",
    }

    missing_from_package = [name for name in expected_symbols if not hasattr(cpl, name)]
    assert missing_from_package == []

    missing_from_registry = sorted(expected_symbols - set(symbol_to_method))
    assert missing_from_registry == []


def test_registry_paths_and_citations_exist() -> None:
    registry = _load_registry()
    citation_ids = set(registry["citations"])
    allowed_status = set(registry["verification_status_values"])

    for method in registry["methods"]:
        assert method["verification_status"] in allowed_status
        for citation_id in method["citation_ids"]:
            assert citation_id in citation_ids
        for rel_path in method["code_paths"] + method["documentation_paths"] + method["tests"]:
            assert (ROOT / rel_path).exists(), rel_path

    for citation_id, citation in registry["citations"].items():
        assert citation["label"]
        assert citation["url"] or citation["doi"], citation_id


def test_science_markdown_matches_registry_identifiers() -> None:
    registry = _load_registry()
    registry_md = (ROOT / "docs" / "science" / "method_registry.md").read_text(
        encoding="utf-8"
    )

    for method in registry["methods"]:
        assert f"`{method['id']}`" in registry_md
    for citation_id in registry["citations"]:
        assert f"`{citation_id}`" in registry_md


def test_science_docs_internal_links_exist() -> None:
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

    for md_path in (ROOT / "docs" / "science").glob("*.md"):
        text = md_path.read_text(encoding="utf-8")
        for target in link_pattern.findall(text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            rel_target = target.split("#", 1)[0]
            assert (md_path.parent / rel_target).exists(), f"{md_path}: {target}"
