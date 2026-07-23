#!/usr/bin/env python3
"""Validate built ChangePointLab distribution artifacts."""

from __future__ import annotations

import argparse
import configparser
import sys
import tarfile
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


BLOCKED_PARTS = {
    "__pycache__",
    "paper",
    "papers",
    "scripts",
    "tests",
    "toolkit",
}
BLOCKED_NAMES = {
    "requirements.txt",
    "setup.py",
}
EXPECTED_SCRIPTS = {
    "cpd": "changepoint_lab.cli.cpd_cli:main",
    "cpd-cli": "changepoint_lab.cli.cpd_cli:main",
    "bocpd-cli": "changepoint_lab.cli.bocpd_cli:main",
    "within-period-cli": "changepoint_lab.algorithms.bayesian.within_period.cli:main",
}


def _load_project(root: Path) -> dict:
    with (root / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]


def _wheel_entries(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return sorted(zf.namelist())


def _sdist_entries(sdist: Path) -> list[str]:
    with tarfile.open(sdist, "r:gz") as tf:
        return sorted(tf.getnames())


def _assert_clean_entries(entries: list[str]) -> None:
    violations: list[str] = []
    for entry in entries:
        parts = Path(entry).parts
        if entry.endswith(".pyc"):
            violations.append(entry)
            continue
        if any(part in BLOCKED_PARTS for part in parts):
            violations.append(entry)
            continue
        if Path(entry).name in BLOCKED_NAMES:
            violations.append(entry)
    if violations:
        formatted = "\n".join(f"  - {path}" for path in violations[:50])
        raise AssertionError(f"unexpected distribution entries:\n{formatted}")


def validate_dist(dist_dir: Path, root: Path) -> None:
    project = _load_project(root)
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1:
        raise AssertionError(f"expected exactly one wheel, found {len(wheels)}")
    if len(sdists) != 1:
        raise AssertionError(f"expected exactly one sdist, found {len(sdists)}")

    wheel_entries = _wheel_entries(wheels[0])
    sdist_entries = _sdist_entries(sdists[0])
    _assert_clean_entries(wheel_entries)
    _assert_clean_entries(sdist_entries)

    dist_info = f"{project['name'].replace('-', '_')}-{project['version']}.dist-info"
    metadata_path = f"{dist_info}/METADATA"
    entry_points_path = f"{dist_info}/entry_points.txt"
    if metadata_path not in wheel_entries:
        raise AssertionError(f"missing {metadata_path}")
    if entry_points_path not in wheel_entries:
        raise AssertionError(f"missing {entry_points_path}")
    if "changepoint_lab/py.typed" not in wheel_entries:
        raise AssertionError("missing changepoint_lab/py.typed")

    with zipfile.ZipFile(wheels[0]) as zf:
        metadata = zf.read(metadata_path).decode("utf-8")
        entry_points = zf.read(entry_points_path).decode("utf-8")

    expected_metadata = [
        f"Name: {project['name']}",
        f"Version: {project['version']}",
        "Requires-Python: >=3.10,<4.0",
    ]
    for line in expected_metadata:
        if line not in metadata:
            raise AssertionError(f"missing wheel metadata line: {line}")

    parser = configparser.ConfigParser()
    parser.read_string(entry_points)
    scripts = dict(parser["console_scripts"])
    for name, target in EXPECTED_SCRIPTS.items():
        if scripts.get(name) != target:
            raise AssertionError(f"missing console script: {name} = {target}")

    if not any(entry.endswith("pyproject.toml") for entry in sdist_entries):
        raise AssertionError("sdist missing pyproject.toml")
    if not any(entry.endswith("src/changepoint_lab/py.typed") for entry in sdist_entries):
        raise AssertionError("sdist missing src/changepoint_lab/py.typed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", type=Path, nargs="?", default=Path("dist"))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    validate_dist(args.dist_dir, root)
    print("distribution artifacts validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
