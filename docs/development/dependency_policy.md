# Dependency policy

Date: 2026-07-23

## Supported compatibility matrix

ChangePointLab supports Python 3.10 through 3.14 for core imports and core
NumPy-based algorithms.

| Layer | Supported range | Validation |
| --- | --- | --- |
| Python | 3.10, 3.11, 3.12, 3.13, 3.14 | CI test matrix |
| NumPy on Python <3.14 | `>=1.21,<2.3` | Python 3.10 minimum job uses NumPy 1.21.6 |
| NumPy on Python >=3.14 | `>=2.3` | Python 3.14 newest job uses current compatible NumPy |
| Plotting extra | Matplotlib `>=3.3` | Full test matrix installs `plot` |
| Data extra | pandas `>=1.5` | Full test matrix installs `data` |

The core package must import and run without Matplotlib or pandas installed.

## Dependency classification

| Package | Classification | Declared in |
| --- | --- | --- |
| NumPy | Runtime core | `[project].dependencies` |
| Matplotlib | Optional plotting | `[project.optional-dependencies].plot` |
| pandas | Optional CSV time-binning/data I/O | `[project.optional-dependencies].data` |
| Sphinx, pdoc, NetworkX | Documentation tooling: Sphinx is the canonical user docs build, pdoc is a secondary API inspection artifact, and NetworkX supports generated documentation utilities | `[tool.poetry.group.docs.dependencies]` |
| pytest | Test execution | `[tool.poetry.group.dev.dependencies]` |
| coverage, pytest-cov, Ruff, Mypy, pydocstyle, types-setuptools, tomli | Development quality gates | `[tool.poetry.group.dev.dependencies]` |
| LibCST | Development migration helper | `[tool.poetry.group.dev.dependencies]` |
| pip-audit, pip-licenses | CI and manual supply-chain checks | `[tool.poetry.group.dev.dependencies]` |

Examples and tutorials may require `plot` and `data`, but those extras must not
be imported by core package initialization.

The Poetry `bench` group is intentionally present but empty until a
benchmark-only dependency is introduced.

## Optional import rules

- Core modules may import NumPy at module import time.
- Plotting modules must import Matplotlib lazily through
  `changepoint_lab._optional.require_matplotlib_pyplot`.
- CSV/data-frame I/O must import pandas lazily through
  `changepoint_lab._optional.require_pandas`.
- Optional dependency failures must mention the missing package and the
  installation extra.

## Local validation commands

```bash
poetry install --with dev,docs --extras "plot data"
poetry check --lock
poetry run ruff check .
poetry run mypy
poetry run pydocstyle src/changepoint_lab
poetry run coverage run -m pytest -m "not slow" tests/unit
poetry run coverage report
poetry run coverage json -o coverage.json
poetry run python scripts/validate_coverage_policy.py coverage.json
poetry run sphinx-build -W --keep-going -b html docs docs/_build/html
poetry run python scripts/validate_docs_links.py
poetry run pytest
poetry build --clean
poetry run python scripts/validate_distribution.py dist
poetry run python scripts/validate_docs_examples.py --dist-dir dist
```

Dependency review commands:

```bash
poetry run pip-licenses --format=plain --fail-on="UNKNOWN;Proprietary" --partial-match
poetry run pip-audit --local --skip-editable --progress-spinner off --desc off --aliases off
```

The vulnerability audit may need network access and current advisory data. It is
part of CI, but it is not part of the normal unit test suite.

## Update policy

1. Add new runtime dependencies only when a NumPy-only implementation is not
   practical for the stable public path.
2. Put feature-specific dependencies behind extras and lazy imports.
3. Update this policy, `pyproject.toml`, CI, and the distribution validator in
   the same change.
4. Run both minimum and newest compatibility jobs before release.
