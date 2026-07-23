# ChangePointLab Contribution Guide

Thank you for your interest in improving ChangePointLab!
This document summarises the development workflow and coding conventions.

## Coding standards

- Format code with **black** (line length 100) and run **ruff** for linting.
- Type check with **mypy**; all public functions must be annotated.
- Follow the naming style: `CamelCase` for classes, `snake_case` for functions and
  variables.

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

We appreciate contributions of all sizes – from typo fixes to new algorithms.
