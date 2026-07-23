# Verification Report: ChangePointLab Reorganization

## Status
Complete as of 2026-07-23.

## Directory Structure & Required Files
- Pass: `src/changepoint_lab/` is the unified package root and includes `_compat.py`, `core/`, `common/`, `algorithms/`, `cli/`, and `py.typed`.
- Pass: legacy root-level algorithm package directories are no longer present. Backward compatibility is handled through `changepoint_lab._compat`.
- Pass: repository-level tests and examples live outside the import package and are not intended wheel contents.

## Public API Exposure & Compatibility
- Pass: top-level imports expose `PELT`, `BOCPD`, `EDivisive`, `HSMM`, `KernelCPD`, `WithinPeriodCPD`, `SDHMM`, and `SDHMMMixVI`.
- Pass: compatibility imports continue to emit `DeprecationWarning`; compatibility tests remain in place for those shims.

## BaseDetector Interface Compliance
- Pass: detector wrappers expose `fit`, `predict`, and `fit_predict` through the unified API.

## Import Hygiene
- Pass: stale `toolkit.api_harmonizer` imports now use `changepoint_lab.common.types.types`.
- Pass: critical undefined-name lint checks pass with `python -m ruff check .`.
- Pass: documentation examples now use unified `changepoint_lab` imports instead of old `bocpd`/`pelt` package examples.

## Tests & Coverage
- Current command: `poetry run pytest`
- Result: 93 passed, 14 expected deprecation warnings.
- Coverage threshold is not enforced yet.

## Typing, Linting, Docstrings
- Developer tools are declared in Poetry dependency groups under `pyproject.toml`.
- Current commands: `poetry run ruff check .`, `poetry run mypy`, and `poetry run pydocstyle src/changepoint_lab`.
- Result: all pass.
- Note: mypy is currently configured for the shared typed core (`src/changepoint_lab/core/datatypes.py` and `src/changepoint_lab/algorithms/_base.py`) while broader algorithm typing debt is handled incrementally.

## Documentation Build
- Pass: Sphinx configuration is present in `docs/conf.py`.
- Current command: `poetry run sphinx-build -b html docs docs/_build/html`.
- Result: pass.

## Packaging
- Pass: `pyproject.toml` is the single metadata source for distribution name `changepoint-lab` and import package `changepoint_lab`.
- Pass: `setup.py` and `requirements.txt` have been removed.
- Current command: `poetry build`.
- Result: wheel and sdist build, manifest comparison, distribution validation, and clean wheel installation pass.

## Deprecation Policy & Examples
- Pass: `CHANGELOG.md` includes deprecation timelines for legacy imports.
- Pass: README and BOCPD/PELT documentation examples have been updated to top-level unified imports.

## Remaining Follow-Up
- Optional: expand mypy coverage beyond the shared typed core.
- Optional: enforce a coverage threshold after raising coverage for CLI and plotting modules.
