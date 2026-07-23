# Scientific Claim Audit

Date: 2026-07-23

This audit records claims found in README, docs, paper material, examples, and
metadata before unsupported prose is rewritten. Classification values are
`verified`, `partially_verified`, `aspirational`, `obsolete`, and `false`.

| ID | Location | Prior claim | Classification | Evidence and correction |
| --- | --- | --- | --- | --- |
| C-001 | `README.md` | Install with `pip install changepoint-lab` and PyPI badge. | false | The current release work is GitHub/Zenodo only; PyPI publishing is out of scope. Rewrite installation to source/GitHub archive paths. |
| C-002 | `README.md` | Migration example imports `from bocpd.bocpd import BOCPD`. | obsolete | The package exposes `changepoint_lab.BOCPD`; legacy `changepoint_lab.bocpd` does not resolve. Replace with current imports. |
| C-003 | `docs/zenodo_metadata.md` | Cite an accompanying JOSS paper. | false | There is no current JOSS submission scope. Replace with Zenodo/CITATION.cff citation guidance only. |
| C-004 | `docs/zenodo_metadata.md` | BOCPD usage runs on `np.random.randn` and exposes `result.changepoints`. | false | Current BOCPD paths support binary/Bernoulli or scalar Poisson-count observations and return `BOCPDResult.cp_prob`, `map_run_length`, and `pred_mean`. Replace usage. |
| C-005 | `docs/comparisons/benchmark_report.md` | Version `v1.0.0`, placeholder images, and superior benchmark numbers. | false | No generated benchmark artifacts were found for these claims. Replace the report with a benchmark-status note and require generated evidence before numeric claims. |
| C-006 | `docs/bocpd_README.md` | Gaussian and Poisson BOCPD extensions are available. | partially_verified | Scalar `PoissonGamma` is now implemented and tested; Gaussian/Student-t BOCPD remains unsupported and should stay out of active documentation. |
| C-007 | `docs/bocpd_README.md` | Hazard extensions significantly improve periodic detection performance. | aspirational | Hazard extensions exist, but no reproducible benchmark evidence is present. Rewrite as available features without performance superiority. |
| C-008 | `docs/bocpd_README.md` | Install package as `bocpd` from PyPI and run `python -m bocpd_cli`. | false | Active distribution name is `changepoint-lab`; PyPI is out of scope; console entry point is `bocpd-cli`. Rewrite. |
| C-009 | `paper.md` | Manuscript-style claims of optimized implementations, broad applications, and better performance. | aspirational | No current publication target or benchmark evidence. Replace root paper text with a scholarly-status note pointing to this audit and registry. |
| C-010 | `docs/bocpd_joss/*` | JOSS checklist, figures, paper, and citation files represent current publication work. | obsolete | JOSS is out of scope. Remove the active JOSS-preparation files after preserving the finding here. |
| C-011 | `examples/multi_method_comparison.py` | Example can be referenced in a JOSS paper. | obsolete | Replace with neutral documentation wording. |
| C-012 | `docs/reproducibility_plan.md` | Future benchmark, publication, Docker, and Zenodo automation APIs are shown as executable examples. | aspirational | These are roadmap ideas, not implemented modules. Rewrite as a non-executable reproducibility roadmap. |
| C-013 | `.zenodo.json` | The release implements count/Gaussian-capable BOCPD and extensive validation utilities across all data modes. | partially_verified | Algorithms exist, but likelihood and validation coverage is uneven. Keep high-level metadata conservative for release records. |
| C-014 | `docs/guide/*`, `docs/parameters/*` | Parameter guidance describes accuracy, robustness, and distribution matching. | partially_verified | Guidance is plausible but not backed by generated benchmark artifacts. Keep qualitative text only when it does not imply measured superiority. |
| C-015 | `docs/science/method_registry.yml` | MySense sliced Poisson process is in project scope. | verified | Martinez-Hernandez and Killick (2024) is recorded, but the registry explicitly marks the method `not_implemented`. |

## Current Corrections Applied

- Source/GitHub installation is now the documented path; PyPI publishing remains out of scope.
- Active citation guidance is Zenodo/CITATION.cff only.
- Unsupported benchmark numbers and placeholder images are removed from active documentation.
- BOCPD active documentation is limited to implemented Beta-Bernoulli and scalar
  Poisson-Gamma paths.
- JOSS preparation material is no longer active documentation.
- The sliced Poisson process is recorded as research scope but not implemented.

## Remaining Follow-up

- Build generated benchmark artifacts before restoring numeric performance claims.
- Add independent scientific oracles before upgrading any method to `verified`.
- Implement or explicitly defer the sliced Poisson process method.
- Replace broad tutorial/parameter heuristics with executable examples and measured evidence.
