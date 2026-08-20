# ChangePointLab Contribution Guide

Thank you for your interest in improving ChangePointLab!
This document summarises the development workflow and coding conventions.

## Coding standards

- Keep lines within 100 characters and run **ruff** for semantic linting.
- Formatting policy is separate from linting; no formatter is configured in Poetry.
- Type check with **mypy**; all public functions must be annotated.
- Follow the naming style: `CamelCase` for classes, `snake_case` for functions and
  variables.

## Local quality checks

Run these commands before opening a pull request:

```bash
poetry run ruff check .
poetry run mypy
poetry run pydocstyle src/changepoint_lab
poetry run pytest
```

## Testing

- Use **pytest** for unit tests. Every feature or bug fix requires accompanying tests.
- Install developer dependencies with `poetry install --with dev,docs --extras "plot data"`.
- Run the full test suite before submitting a pull request: `poetry run pytest`.

## BaseDetector interface

All detectors must inherit from `BaseDetector` in `changepoint_lab.algorithms._base`
and implement `fit`, `predict`, and `fit_predict`. Returned results should be
`ChangePointResult` instances with sorted indices.

## Shared utilities

Use this table to decide where helpers belong:

| Location | Purpose |
|---------|---------|
| `core/` | Types, validation and metrics shared across algorithms |
| `common/` | Generic utilities (logging, random seeds) |
| local submodule | Algorithm-specific helpers |

## Pull requests

1. Create a feature branch off the main repository.
2. Run formatting, type checking, and tests through Poetry.
3. Ensure documentation and examples use `from changepoint_lab import ...` imports.
4. Submit the pull request with a clear description of the change.

## Issues and security

- Use the issue templates so reports include enough reproduction, environment,
  and validation context.
- Use the scientific validation template for numerical or paper-replication
  concerns.
- Report suspected vulnerabilities through GitHub private vulnerability
  reporting, not public issues.
- Follow `CODE_OF_CONDUCT.md` in project discussions and reviews.

We appreciate contributions of all sizes - from typo fixes to new algorithms.
