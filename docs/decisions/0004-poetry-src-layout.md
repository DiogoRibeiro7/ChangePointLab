# 0004: Poetry packaging and src layout

Date: 2026-07-23

## Decision

ChangePointLab uses `pyproject.toml` as the single source of package metadata,
Poetry/poetry-core for builds, distribution name `changepoint-lab`, and import
package `changepoint_lab` under `src/changepoint_lab`.

## Context

The repository previously carried multiple packaging paths: setuptools build
metadata, `setup.py`, `requirements.txt`, and console scripts targeting the
separate `toolkit` package. That made dependency drift and accidental
repository-root imports likely during local tests.

## Consequences

- Editable installs and CI use `poetry install`.
- Release artifacts are produced with `poetry build`.
- Console scripts resolve to package-local modules under `changepoint_lab`.
- Repository tests and runnable examples live outside the import package and
  are excluded from the wheel.
- No scientific defaults, numerical tolerances, or estimator return values were
  intentionally changed by this packaging decision.
