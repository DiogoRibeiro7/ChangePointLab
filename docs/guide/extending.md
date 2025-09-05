# Extending ChangePointLab

This guide outlines the minimal steps to add a new detector to
ChangePointLab and expose it through the public API.

## 1. Create the detector

1. Place the implementation under an appropriate subfolder of
   `changepoint_lab/algorithms/`.
2. Inherit from `BaseDetector` defined in `algorithms/_base.py`.
3. Return a `ChangePointResult` from `predict` and support `fit`, `predict`
   and `fit_predict`.

## 2. Register public symbols

Export the detector class and any helper types in the module's `__all__` and
re-export the detector in `changepoint_lab/__init__.py` to make
`from changepoint_lab import YourDetector` work.

## 3. Shared utilities

Use this decision table to place helpers:

| Where? | Use for |
|--------|---------|
| `core/` | Types, validation, metrics shared across algorithms |
| `common/` | Generic helpers such as logging or RNG |
| local module | Implementation details specific to the algorithm |

## 4. Documentation and tests

1. Add an API page under `docs/api/` using autodoc.
2. Update `docs/guide/index.md` or other tutorials if relevant.
3. Include unit tests under `tests/` exercising the new detector.

Following this workflow keeps the codebase consistent and ensures new
algorithms integrate smoothly with the rest of ChangePointLab.
