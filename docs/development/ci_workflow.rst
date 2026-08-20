CI Workflow
===========

The CI workflow is a regression gate for packaging, public API contracts,
documentation builds, and supported dependency ranges. It is not a statistical
proof of method correctness; scientific claims still require the independent
oracles and traceability records listed in ``docs/science/``.

Quality
-------

The quality job runs the configured static checks, builds canonical Sphinx
documentation with warnings as errors, validates local documentation links,
builds distribution artifacts, and executes selected documentation examples
against the built wheel.

This job proves that the configured public static-check surface, documentation
source tree, and package artifacts are internally consistent. It does not prove
that all source files are exhaustively type checked or that unmarked examples
execute.

Tests
-----

The test matrix runs non-slow unit tests across Python 3.10, 3.11, 3.12, 3.13,
and 3.14 with optional plotting and data dependencies installed. Coverage is
measured against ``src/changepoint_lab`` only. CI enforces a 65% overall source
coverage floor and stricter floors for selected core scientific areas:

.. list-table::
   :header-rows: 1

   * - Area
     - Floor
   * - Core contracts
     - 90%
   * - Optimization methods
     - 90%
   * - Nonparametric methods
     - 90%
   * - Point-process methods
     - 85%
   * - BOCPD methods
     - 75%

The floors are measured from the current accepted baseline and should be
ratcheted upward when new tests land. They prevent silent collapse, but they do
not replace targeted oracle tests for numerical or statistical behavior.

Supply Chain
------------

The supply-chain job installs the development and documentation environment,
runs ``pip-audit`` against installed third-party distributions, and runs a
license metadata audit that fails on unknown or proprietary license strings.

This job catches known vulnerability advisories and missing license metadata at
CI time. It does not prove that a dependency is free of undisclosed
vulnerabilities, and it intentionally treats development-only tooling
separately from runtime package metadata.

Documentation
-------------

The documentation job builds the canonical Sphinx HTML artifact and the
secondary pdoc API inspection artifact. Sphinx is the source of truth for user
documentation.

Cross-Platform
--------------

The cross-platform job runs unit tests on Windows and macOS using Python 3.11.
It catches common path, shell, and platform assumptions. It is not a full Python
version matrix on every operating system.

Compatibility
-------------

The compatibility job builds a wheel and installs it into a temporary
environment with only NumPy. It runs explicit oldest/newest NumPy checks:

.. list-table::
   :header-rows: 1

   * - Label
     - Python
     - NumPy
   * - oldest
     - 3.10
     - 1.21.6
   * - newest
     - 3.14
     - >=2.3

This job proves that core imports and a small PELT smoke path work from the
wheel without Matplotlib or pandas. It does not cover optional extras.

Local Commands
--------------

Run the main CI checks locally with:

.. code-block:: bash

   poetry check --lock
   poetry run ruff check .
   poetry run mypy
   poetry run pydocstyle src/changepoint_lab
   poetry run coverage run -m pytest -m "not slow" tests/unit
   poetry run coverage report
   poetry run coverage json -o coverage.json
   poetry run python scripts/validate_coverage_policy.py coverage.json
   poetry build --clean
   poetry run python scripts/validate_distribution.py dist
   poetry run python scripts/validate_docs_examples.py --dist-dir dist
   poetry run python scripts/validate_docs_links.py
   poetry run pip-audit --local --skip-editable --progress-spinner off --desc off --aliases off
   poetry run pip-licenses --format=plain --fail-on="UNKNOWN;Proprietary" --partial-match
