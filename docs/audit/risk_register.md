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
| R-002 | Critical | Scientific correctness | `src/changepoint_lab/algorithms/bayesian/bocpd/likelihoods.py` has TODO placeholders for `PoissonGamma.predictive_prob`, `PoissonGamma.update_*`, and `GaussianNIW` methods. | Documentation or API may imply count/Gaussian BOCPD support that is not implemented. | Mark unsupported likelihoods explicitly or implement with independent oracle tests. |
| R-003 | Critical | Scientific correctness | BOCPD wrapper extracts changepoints as `cp_prob > 0.5`; prior audit notes scaled probabilities. | Reported probabilities and detected events may not correspond to canonical posterior semantics. | Freeze current behavior, derive canonical formulas, add calibration/oracle tests. |
| R-004 | High | API correctness | `changepoint_lab.bocpd` is used in `paper.md`, but `getattr(changepoint_lab, "bocpd")` raises `AttributeError`. | Published examples can fail for users. | Update docs after preserving stale claim in claim audit; add compatibility test or documented removal. |
| R-006 | High | Scientific correctness | Within-period RJMCMC uses `np.random.seed` and `random.seed`; proposals rely on module-level `random`. | Hidden global RNG state can break reproducibility and composition. | Characterize current seeded behavior, then move to explicit generator/state objects. |
| R-007 | High | Scientific correctness | Within-period prior/proposal details are not independently verified; prior audit identified Poisson lambda and reverse-proposal issues; a tiny `N=4, l=1` seeded fixture raises `ValueError`. | Sampler may not target the stated posterior distribution. | Build paper-derived and brute-force small-state oracles before changing behavior. |
| R-008 | High | Documentation | `docs/comparisons/benchmark_report.md` previously contained placeholder image URLs, version `v1.0.0`, and strong superiority claims. | Users may rely on unsupported performance and adoption claims if old claims are restored without evidence. | Keep benchmark claims blocked until generated artifacts exist. |
| R-009 | High | Release metadata | `docs/zenodo_metadata.md` previously said to cite an accompanying JOSS paper and had "Zenodo DOI assigned on release" prose. | Citation guidance can drift from current "Zenodo only, no JOSS" release scope. | Keep release metadata aligned before each Zenodo release. |
| R-012 | Medium | Type/quality | Mypy is scoped to two files only; Ruff is configured for critical errors only. | Many interface and style defects are outside current gates. | Broaden gates after correctness baseline exists. |
| R-013 | Medium | Public API | `src/changepoint_lab/common/api_harmonizer.py` exposes dictionary-heavy adapter metadata and remains internal production code. | Public contract is unclear and conflicts with typed result-object goal if exposed later. | Keep it out of documented stable API or replace it with typed result-object interfaces. |
| R-014 | Medium | Documentation | Root `README.md` previously contained `from bocpd.bocpd import BOCPD`. | Quickstart/migration examples can drift after package reorganization. | Keep executable-documentation checks aligned with current imports. |
| R-015 | Medium | Scientific traceability | `docs/science/method_registry.yml` now maps methods to sources, deviations, and tests, but several methods remain below `verified`. | Scientific claims can still overstate evidence if verification status is ignored. | Add independent oracles before marking any method `verified`. |
| R-016 | Low | Repository hygiene | Empty marker files are expected, but package `__init__.py` files are inconsistent about exports. | Discoverability and API boundaries are unclear. | Normalize package exports after public API contract decision. |

## Resolved or Partially Resolved Findings

| ID | Status | Evidence |
| --- | --- | --- |
| RR-001 | Resolved before this audit | `pyproject.toml`, runtime `__version__`, CFF, and `.zenodo.json` are aligned at `0.1.5`. |
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

## Blockers Before External Scientific Readiness

1. Fix or document broken public compatibility paths such as legacy `bocpd`.
2. Establish method-to-source traceability and claim audit.
3. Freeze current behavior with golden characterization tests before correcting scientific algorithms.
4. Add independent oracles for PELT, BOCPD, within-period RJMCMC, KCP/RFF, E-Divisive, HSMM, and SD-HMM.
5. Replace unsupported docs and paper claims with executable evidence.
