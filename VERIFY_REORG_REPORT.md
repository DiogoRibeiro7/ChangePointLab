# Verification Report: ChangePointLab Reorganization

## Status
Complete as of 2026-07-23.

## Directory Structure & Required Files
- Pass: `changepoint_lab/` is the unified package root and includes `_compat.py`, `core/`, `common/`, `algorithms/`, and `py.typed`.
- Pass: legacy root-level algorithm package directories are no longer present. Backward compatibility is handled through `changepoint_lab._compat`.

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
- Pass: `python -m pytest --cov=changepoint_lab --cov=toolkit`
- Result: 80 passed, 14 expected deprecation warnings.
- Coverage summary: 49% total line coverage reported. No coverage threshold is enforced yet.

## Typing, Linting, Docstrings
- Pass: developer tools are declared in `pyproject.toml` under `[project.optional-dependencies].dev`.
- Pass: `python -m ruff check .`
- Pass: `python -m mypy`
- Pass: `python -m pydocstyle changepoint_lab`
- Note: mypy is currently configured for the shared typed core (`changepoint_lab/core/datatypes.py` and `changepoint_lab/algorithms/_base.py`) while broader algorithm typing debt is handled incrementally.

## Documentation Build
- Pass: Sphinx configuration is present in `docs/conf.py`.
- Pass: `python -m sphinx -b html docs docs/_build/html`

## Packaging
- Pass: `python -m build`
- Pass: `setup.py` is now a minimal pyproject-compatible shim, avoiding duplicate package metadata.

## Deprecation Policy & Examples
- Pass: `CHANGELOG.md` includes deprecation timelines for legacy imports.
- Pass: README and BOCPD/PELT documentation examples have been updated to top-level unified imports.

## Remaining Follow-Up
- Optional: expand mypy coverage beyond the shared typed core.
- Optional: enforce a coverage threshold after raising coverage for CLI, plotting, and example modules.
