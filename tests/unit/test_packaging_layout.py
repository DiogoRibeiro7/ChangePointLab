from __future__ import annotations

import importlib.util
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]


def test_import_package_uses_src_layout() -> None:
    assert (ROOT / "src" / "changepoint_lab" / "__init__.py").exists()
    assert (ROOT / "src" / "changepoint_lab" / "py.typed").exists()
    assert not (ROOT / "changepoint_lab").exists()
    assert not (ROOT / "toolkit").exists()


def test_pyproject_declares_poetry_src_package() -> None:
    with (ROOT / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)

    assert data["project"]["name"] == "changepoint-lab"
    assert data["build-system"]["build-backend"] == "poetry.core.masonry.api"
    assert data["tool"]["poetry"]["packages"] == [
        {"include": "changepoint_lab", "from": "src"}
    ]
    assert "pythonpath" not in data["tool"]["pytest"]["ini_options"]


def test_legacy_toolkit_package_is_not_importable() -> None:
    assert importlib.util.find_spec("toolkit") is None
