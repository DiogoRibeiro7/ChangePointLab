# Risk Register

Date: 2026-07-23
Scope: repository forensic audit only

Severity levels:

- Critical: likely to invalidate scientific results or block basic package use.
- High: public API, packaging, or documentation issue likely to mislead users.
- Medium: maintainability, coverage, or process risk that can mask defects.
- Low: cleanup or polish issue.

## Open Risks

| ID | Severity | Category | Evidence | Risk | Required next action |
| --- | --- | --- | --- | --- | --- |
| R-002 | Medium | Scientific correctness | `src/changepoint_lab/algorithms/bayesian/bocpd/likelihoods.py` still contains `GaussianNIW` placeholder methods. | Users may infer Gaussian BOCPD support from internal class names if not kept out of public docs. | Keep Gaussian/Student-t BOCPD undocumented as supported until implemented with independent oracle tests. |
| R-004 | High | API correctness | `changepoint_lab.bocpd` is used in `paper.md`, but `getattr(changepoint_lab, "bocpd")` raises `AttributeError`. | Published examples can fail for users. | Update docs after preserving stale claim in claim audit; add compatibility test or documented removal. |
| R-008 | High | Documentation | `docs/comparisons/benchmark_report.md` previously contained placeholder image URLs, version `v1.0.0`, and strong superiority claims. | Users may rely on unsupported performance and adoption claims if old claims are restored without evidence. | Keep benchmark claims blocked until generated artifacts exist. |
| R-009 | High | Release metadata | `docs/zenodo_metadata.md` previously said to cite an accompanying JOSS paper and had "Zenodo DOI assigned on release" prose. | Citation guidance can drift from current "Zenodo only, no JOSS" release scope. | Keep release metadata aligned before each Zenodo release. |
| R-012 | Medium | Type/quality | Mypy is scoped to two files only; Ruff is configured for critical errors only. | Many interface and style defects are outside current gates. | Broaden gates after correctness baseline exists. |
| R-013 | Medium | Public API | `src/changepoint_lab/common/api_harmonizer.py` exposes dictionary-heavy adapter metadata and remains internal production code. | Public contract is unclear and conflicts with typed result-object goal if exposed later. | Keep it out of documented stable API or replace it with typed result-object interfaces. |
| R-014 | Medium | Documentation | Root `README.md` previously contained `from bocpd.bocpd import BOCPD`. | Quickstart/migration examples can drift after package reorganization. | Keep executable-documentation checks aligned with current imports. |
| R-015 | Medium | Scientific traceability | `docs/science/method_registry.yml` now maps methods to sources, deviations, and tests, but several methods remain below `verified`. | Scientific claims can still overstate evidence if verification status is ignored. | Add independent oracles before marking any method `verified`. |
| R-016 | Low | Repository hygiene | Empty marker files are expected, but package `__init__.py` files are inconsistent about exports. | Discoverability and API boundaries are unclear. | Normalize package exports after public API contract decision. |
| R-017 | Medium | Scientific reproducibility | Taylor et al. case-study sensor data are not bundled; `scripts/run_within_period_reproduction.py` generates synthetic analogues and records discrepancies. | Users may mistake synthetic MySense outputs for recreation of the proprietary case-study figures. | Keep paper-consistent and MySense-extension artifacts separated, and add real cached data only with license and checksum documentation. |
| R-018 | Medium | Scientific reproducibility | Sliced Poisson tests cover analytical and simulated cases, but the Howz data and supplementary code parity are not bundled. | Users may overinterpret the implementation as full reproduction of Martínez-Hernández and Killick (2024). | Keep the method marked `partially_verified`; add licensed reference data or supplementary-code parity only when available. |

## Resolved or Partially Resolved Findings

| ID | Status | Evidence |
| --- | --- | --- |
| RR-001 | Resolved before this audit | `pyproject.toml`, runtime `__version__`, CFF, and `.zenodo.json` are aligned at `0.1.11`. |
| RR-002 | Resolved before this audit | `setup.py` now delegates to `setup()` and no longer duplicates metadata. |
| RR-003 | Partially resolved | CI now has a `quality` job and builds distributions, but type/lint scope remains deliberately narrow. |
| RR-004 | Partially resolved | Runtime dependencies are declared in `pyproject.toml`; `requirements.txt` still duplicates them. |
| RR-005 | Partially resolved | Scientific method registry and claim audit now exist; independent scientific oracles are still pending. |
| RR-006 | Partially resolved | Unsupported active benchmark/JOSS/PyPI claims were rewritten or removed from the main documentation path. |
| RR-007 | Resolved | Root `README.md` no longer contains the stale `from bocpd.bocpd import BOCPD` import. |
| RR-008 | Partially resolved | Baseline fixtures and tests now capture current outputs, warnings, exceptions, and independent tiny oracles before corrective refactoring. |
| RR-009 | Resolved | Importable package code now lives under `src/changepoint_lab`, `setup.py` and `requirements.txt` were removed, and dependency metadata is centralized in Poetry-managed `pyproject.toml`. |
| RR-010 | Resolved | Runnable package examples and package-local KCP tests were moved out of the wheel package tree. |
| RR-011 | Resolved | Former `toolkit` entry points now target package-local CLI modules under `changepoint_lab.cli`. |
| RR-012 | Resolved | Core runtime dependencies are now NumPy-only; Matplotlib and pandas are optional extras with lazy import errors and CI coverage. |
| RR-013 | Resolved | `KernelCPD.fit_predict(...)` now builds a kernel prefix and returns a typed result instead of raising `AttributeError`. |
| RR-014 | Resolved | Top-level public API tests now exercise every stable estimator wrapper with a real tiny input. |
| RR-015 | Resolved | HSMM wrapper changepoints now come from nonzero sparse duration-end markers and match the tiny Viterbi oracle. |
| RR-016 | Resolved | `SDHMMMixVI.fit_predict(...)` now updates parameters through a mutable local copy and completes the minimal fixture. |
| RR-017 | Resolved | Exact and RFF KCP backtracking now drops terminal endpoint `n` and returns validated right-exclusive interior boundaries; the tiny kernel oracle now reports `[2]` with edges `[0, 2, 4]`. |
| RR-018 | Resolved | Production stochastic paths now use owned `numpy.random.Generator` streams instead of module-level RNG state; stochastic typed results expose provenance. |
| RR-019 | Partially resolved | Within-period RJMCMC now uses an exact enumerated proposal kernel, includes `pois_lambda` in segment-count posterior comparisons, rejects invalid proposal weights, forbids singleton circular states, and has tiny-state detailed-balance/empirical-stationary tests. Full replication against published examples remains pending. |
| RR-020 | Partially resolved | A one-command within-period reproduction workflow now generates paper-style synthetic scenarios, MySense-style synthetic sensor outputs, posterior summaries, diagnostics, prior sensitivity, and discrepancy notes; notebook execution is covered by tests. |
| RR-021 | Partially resolved | Sliced Poisson process changepoint detection is now exposed as a dedicated API with B-spline IHPP segment costs, PELT optimization, exposure intervals, independent marked fitting, simulations, diagnostics, and focused tests. |
| RR-022 | Partially resolved | BOCPD now defaults to the unscaled run-length posterior, moves changepoint alerts into `BOCPDAlertConfig`, deprecates `cp_scale`, and includes hand-computed, independent-recursion, normalization, alert-policy, and approximation diagnostics tests. Broader likelihood support remains pending. |
| RR-023 | Partially resolved | BOCPD now accepts explicit likelihood instances, preserves them across `reset()`, implements scalar `PoissonGamma`, supports `update_many`, missing-observation transitions, and checkpoint/resume state dictionaries. Gaussian/Student-t BOCPD remains unsupported. |

## Blockers Before External Scientific Readiness

1. Fix or document broken public compatibility paths such as legacy `bocpd`.
2. Establish method-to-source traceability and claim audit.
3. Freeze current behavior with golden characterization tests before correcting scientific algorithms.
4. Add independent oracles for PELT, BOCPD, within-period RJMCMC, KCP/RFF, E-Divisive, HSMM, and SD-HMM.
5. Replace unsupported docs and paper claims with executable evidence.
