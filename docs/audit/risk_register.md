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
| R-001 | Critical | API correctness | `KernelCPD(penalty=1.0).fit_predict(...)` failed with `AttributeError: 'tuple' object has no attribute 'K'` | Public top-level estimator is exported but broken on its default path. | Add characterization test, decide expected wrapper contract, then fix in the KernelCPD workstream. |
| R-002 | Critical | Scientific correctness | `changepoint_lab/algorithms/bayesian/bocpd/likelihoods.py` has TODO placeholders for `PoissonGamma.predictive_prob`, `PoissonGamma.update_*`, and `GaussianNIW` methods. | Documentation or API may imply count/Gaussian BOCPD support that is not implemented. | Mark unsupported likelihoods explicitly or implement with independent oracle tests. |
| R-003 | Critical | Scientific correctness | BOCPD wrapper extracts changepoints as `cp_prob > 0.5`; prior audit notes scaled probabilities. | Reported probabilities and detected events may not correspond to canonical posterior semantics. | Freeze current behavior, derive canonical formulas, add calibration/oracle tests. |
| R-004 | High | API correctness | `changepoint_lab.bocpd` is used in `paper.md`, but `getattr(changepoint_lab, "bocpd")` raises `AttributeError`. | Published examples can fail for users. | Update docs after preserving stale claim in claim audit; add compatibility test or documented removal. |
| R-005 | High | Packaging/API | Packaged modules `changepoint_lab.examples.edivisive_example`, `hsmm_example`, `kcp_example`, and `kcp_rff_example` timed out on import; `sdhmm_mix_vi_example` failed on import. | Installed package contains importable modules with top-level execution or errors. | Move runnable example code behind `main()` guards or remove examples from package distribution. |
| R-006 | High | Scientific correctness | Within-period RJMCMC uses `np.random.seed` and `random.seed`; proposals rely on module-level `random`. | Hidden global RNG state can break reproducibility and composition. | Characterize current seeded behavior, then move to explicit generator/state objects. |
| R-007 | High | Scientific correctness | Within-period prior/proposal details are not independently verified; prior audit identified Poisson lambda and reverse-proposal issues. | Sampler may not target the stated posterior distribution. | Build paper-derived and brute-force small-state oracles before changing behavior. |
| R-008 | High | Documentation | `docs/comparisons/benchmark_report.md` contains placeholder image URLs, version `v1.0.0`, and strong superiority claims. | Users may rely on unsupported performance and adoption claims. | Claim audit must classify unsupported claims and remove or replace them with generated evidence. |
| R-009 | High | Release metadata | `docs/zenodo_metadata.md` says to cite an accompanying JOSS paper and has "Zenodo DOI assigned on release" prose. | Citation guidance conflicts with current "Zenodo only, no JOSS" release scope. | Update scholarly metadata docs after claim audit. |
| R-010 | Medium | Packaging | `requirements.txt` duplicates runtime dependencies already declared in `pyproject.toml`. | Dependency drift can recur. | Decide whether to remove it or generate it from project metadata in dependency-audit work. |
| R-011 | Medium | Tests | Top-level public API tests mostly verify exports and deprecation warnings, not all estimator behavior. | Broken public wrappers can pass CI. | Add smoke and characterization tests for all exported stable estimators. |
| R-012 | Medium | Type/quality | Mypy is scoped to two files only; Ruff is configured for critical errors only. | Many interface and style defects are outside current gates. | Broaden gates after correctness baseline exists. |
| R-013 | Medium | Public API | `toolkit/api_harmonizer.py` exposes dictionary-heavy adapter metadata and is packaged as production code. | Public contract is unclear and conflicts with typed result-object goal. | Decide whether toolkit is stable API, compatibility layer, or internal CLI support. |
| R-014 | Medium | Documentation | Root `README.md` still contains `from bocpd.bocpd import BOCPD`. | Quickstart/migration story is inconsistent. | Replace stale example during executable-documentation work. |
| R-015 | Medium | Scientific traceability | No machine-readable method registry maps methods to sources, deviations, and tests. | Scientific claims are hard to verify and maintain. | Implement method registry and claim audit before scientific changes. |
| R-016 | Low | Repository hygiene | Empty marker files are expected, but package `__init__.py` files are inconsistent about exports. | Discoverability and API boundaries are unclear. | Normalize package exports after public API contract decision. |

## Resolved or Partially Resolved Findings

| ID | Status | Evidence |
| --- | --- | --- |
| RR-001 | Resolved before this audit | `pyproject.toml`, runtime `__version__`, CFF, and `.zenodo.json` are aligned at `0.1.2`. |
| RR-002 | Resolved before this audit | `setup.py` now delegates to `setup()` and no longer duplicates metadata. |
| RR-003 | Partially resolved | CI now has a `quality` job and builds distributions, but type/lint scope remains deliberately narrow. |
| RR-004 | Partially resolved | Runtime dependencies are declared in `pyproject.toml`; `requirements.txt` still duplicates them. |

## Blockers Before External Scientific Readiness

1. Fix or document broken public estimator paths (`KernelCPD`, legacy `bocpd`, packaged examples).
2. Establish method-to-source traceability and claim audit.
3. Freeze current behavior with golden characterization tests before correcting scientific algorithms.
4. Add independent oracles for PELT, BOCPD, within-period RJMCMC, KCP/RFF, E-Divisive, HSMM, and SD-HMM.
5. Replace unsupported docs and paper claims with executable evidence.
