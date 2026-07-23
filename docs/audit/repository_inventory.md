# Repository Inventory

Date: 2026-07-23
Branch: `main`
Baseline commit: `13e99810bccab2e869dc295676d7a51eb0c63c9e`

This inventory classifies the tracked repository state before scientific or API
modernization work. It is intentionally descriptive: no production behavior was
changed during this audit.

## Summary

- Tracked files: 179 (`git ls-files`)
- Python files in tracked tree: 111
- Markdown files in tracked tree: 44
- reStructuredText files in tracked tree: 10
- Notebooks in tracked tree: 1
- Importable `changepoint_lab` modules discovered by `pkgutil.walk_packages`: 64
- Console entry points declared in `pyproject.toml`: 4
- Empty tracked marker files: 15, all package/type marker files
- Local workflow notes are ignored by `.git/info/exclude` and not tracked

## File Classification

Every tracked file is classified below.

### Production Package (55)

- `changepoint_lab/__init__.py`
- `changepoint_lab/algorithms/__init__.py`
- `changepoint_lab/algorithms/_base.py`
- `changepoint_lab/algorithms/bayesian/__init__.py`
- `changepoint_lab/algorithms/bayesian/bocpd/__init__.py`
- `changepoint_lab/algorithms/bayesian/bocpd/core.py`
- `changepoint_lab/algorithms/bayesian/bocpd/likelihoods.py`
- `changepoint_lab/algorithms/bayesian/bocpd/plotting.py`
- `changepoint_lab/algorithms/bayesian/bocpd/validation.py`
- `changepoint_lab/algorithms/bayesian/within_period/__init__.py`
- `changepoint_lab/algorithms/bayesian/within_period/anchor_utils.py`
- `changepoint_lab/algorithms/bayesian/within_period/posterior_predictive.py`
- `changepoint_lab/algorithms/bayesian/within_period/samplers/__init__.py`
- `changepoint_lab/algorithms/bayesian/within_period/samplers/tempering.py`
- `changepoint_lab/algorithms/bayesian/within_period/within_period_cpd.py`
- `changepoint_lab/algorithms/kernel/__init__.py`
- `changepoint_lab/algorithms/kernel/bandwidth_cv.py`
- `changepoint_lab/algorithms/kernel/kcp.py`
- `changepoint_lab/algorithms/kernel/kcp_core.py`
- `changepoint_lab/algorithms/kernel/kcp_rff.py`
- `changepoint_lab/algorithms/kernel/rff_variants.py`
- `changepoint_lab/algorithms/nonparametric/__init__.py`
- `changepoint_lab/algorithms/nonparametric/edivisive.py`
- `changepoint_lab/algorithms/nonparametric/edivisive_core.py`
- `changepoint_lab/algorithms/optimization/__init__.py`
- `changepoint_lab/algorithms/optimization/cost_functions.py`
- `changepoint_lab/algorithms/optimization/pelt.py`
- `changepoint_lab/algorithms/state_space/__init__.py`
- `changepoint_lab/algorithms/state_space/emissions/__init__.py`
- `changepoint_lab/algorithms/state_space/emissions/ar_emissions.py`
- `changepoint_lab/algorithms/state_space/emissions/gaussian_diag.py`
- `changepoint_lab/algorithms/state_space/emissions/gaussian_full.py`
- `changepoint_lab/algorithms/state_space/hsmm.py`
- `changepoint_lab/algorithms/state_space/hsmm_core.py`
- `changepoint_lab/algorithms/state_space/sdhmm.py`
- `changepoint_lab/algorithms/state_space/sdhmm_mix_vi.py`
- `changepoint_lab/common/__init__.py`
- `changepoint_lab/common/diagnostics/__init__.py`
- `changepoint_lab/common/diagnostics/diagnostics.py`
- `changepoint_lab/common/io/__init__.py`
- `changepoint_lab/common/io/data_loader.py`
- `changepoint_lab/common/io/io_utils.py`
- `changepoint_lab/common/plotting/__init__.py`
- `changepoint_lab/common/plotting/edivisive_plotting.py`
- `changepoint_lab/common/plotting/kcp_plotting.py`
- `changepoint_lab/common/plotting/plotting_helpers.py`
- `changepoint_lab/common/types/__init__.py`
- `changepoint_lab/common/types/types.py`
- `changepoint_lab/common/utils/__init__.py`
- `changepoint_lab/common/utils/utils.py`
- `changepoint_lab/core/__init__.py`
- `changepoint_lab/core/datatypes.py`
- `changepoint_lab/py.typed`
- `toolkit/__init__.py`
- `toolkit/api_harmonizer.py`

### Compatibility Shim (1)

- `changepoint_lab/_compat.py`

### CLI (4)

- `changepoint_lab/algorithms/bayesian/within_period/cli.py`
- `changepoint_lab/cli/__init__.py`
- `changepoint_lab/cli/bocpd_cli.py`
- `toolkit/cpd_cli.py`

### Tests (23)

- `changepoint_lab/tests/test_kcp/__init__.py`
- `changepoint_lab/tests/test_kcp/test_bandwidth_cv.py`
- `changepoint_lab/tests/test_kcp/test_rff_variants.py`
- `conftest.py`
- `tests/integration/test_cli_integration.py`
- `tests/integration/test_methods_interoperability.py`
- `tests/test_public_api.py`
- `tests/unit/test_algorithm_parity.py`
- `tests/unit/test_bocpd_edge_cases.py`
- `tests/unit/test_edivisive_memmap.py`
- `tests/unit/test_hsmm/__init__.py`
- `tests/unit/test_hsmm/test_emission_models.py`
- `tests/unit/test_hsmm_duration_cache.py`
- `tests/unit/test_interoperability.py`
- `tests/unit/test_kcp_segment_bounds.py`
- `tests/unit/test_method_comparison.py`
- `tests/unit/test_numerical_stability.py`
- `tests/unit/test_package.py`
- `tests/unit/test_pelt_result_fields.py`
- `tests/unit/test_performance.py`
- `tests/unit/test_plotting_smoke.py`
- `toolkit/tests/__init__.py`
- `toolkit/tests/test_cli_wrapper.py`

### Examples (25)

- `changepoint_lab/examples/__init__.py`
- `changepoint_lab/examples/bocpd_examples.py`
- `changepoint_lab/examples/bocpd_notebook.md`
- `changepoint_lab/examples/edivisive_example.py`
- `changepoint_lab/examples/hsmm_example.py`
- `changepoint_lab/examples/kcp_example.py`
- `changepoint_lab/examples/kcp_rff_example.py`
- `changepoint_lab/examples/sdhmm_example.py`
- `changepoint_lab/examples/sdhmm_mix_vi_example.py`
- `changepoint_lab/examples/within_period_example.py`
- `examples/bocpd_activity_monitoring.py`
- `examples/bocpd_binary_stream.py`
- `examples/comparison_helpers.py`
- `examples/edivisive_climate_data.py`
- `examples/hsmm_medical_monitoring.py`
- `examples/integration/__init__.py`
- `examples/integration/ensemble_detection.py`
- `examples/integration/hierarchical_detection.py`
- `examples/integration/multivariate_integration.py`
- `examples/integration/online_offline_hybrid.py`
- `examples/integration/two_stage_detection.py`
- `examples/multi_method_comparison.py`
- `examples/pelt_financial_time_series.py`
- `examples/quickstart.py`
- `examples/sdhmm_microbiome_analysis.py`

### Notebook (1)

- `docs/notebooks/multi_method_comparison.ipynb`

### Scripts (3)

- `scripts/doc_generator.py`
- `scripts/generate_figures.py`
- `scripts/rewrite_cpd_imports.py`

### Paper and Scholarly Material (8)

- `docs/bocpd_joss/CITATION.cff`
- `docs/bocpd_joss/bocpd_diagram.md`
- `docs/bocpd_joss/joss_figure_png.md`
- `docs/bocpd_joss/joss_review.md`
- `docs/bocpd_joss/paper.bib`
- `docs/bocpd_joss/paper.md`
- `paper.bib`
- `paper.md`

### Documentation (45)

- `changepoint_lab/docs/bocpd_guide.md`
- `docs/api/bocpd.rst`
- `docs/api/datatypes.rst`
- `docs/api/edivisive.rst`
- `docs/api/hsmm.rst`
- `docs/api/index.rst`
- `docs/api/kcp.rst`
- `docs/api/pelt.rst`
- `docs/api/sdhmm.rst`
- `docs/api/within_period.rst`
- `docs/architecture/index.md`
- `docs/bocpd_README.md`
- `docs/comparisons/benchmark_report.md`
- `docs/comparisons/binary_data_comparison.md`
- `docs/comparisons/compositional_data_comparison.md`
- `docs/comparisons/computational_efficiency.md`
- `docs/comparisons/continuous_data_comparison.md`
- `docs/comparisons/online_vs_offline.md`
- `docs/comparisons/robustness_comparison.md`
- `docs/conf.py`
- `docs/cpd.md`
- `docs/guide/bayesian_methods.md`
- `docs/guide/choosing_methods.md`
- `docs/guide/extending.md`
- `docs/guide/index.md`
- `docs/guide/nonparametric_methods.md`
- `docs/guide/optimization_methods.md`
- `docs/guide/state_space_methods.md`
- `docs/guide/visualization.md`
- `docs/index.rst`
- `docs/migration/bayesian_blocks.md`
- `docs/parameters/bocpd_parameters.md`
- `docs/parameters/edivisive_parameters.md`
- `docs/parameters/hsmm_parameters.md`
- `docs/parameters/pelt_parameters.md`
- `docs/parameters/sdhmm_parameters.md`
- `docs/parameters/within_period_parameters.md`
- `docs/reproducibility_plan.md`
- `docs/tutorials/environmental_science_tutorial.md`
- `docs/tutorials/finance_tutorial.md`
- `docs/tutorials/getting_started_tutorial.md`
- `docs/tutorials/healthcare_tutorial.md`
- `docs/tutorials/industrial_monitoring_tutorial.md`
- `docs/tutorials/iot_tutorial.md`
- `docs/zenodo_metadata.md`

### Metadata and Packaging (10)

- `.zenodo.json`
- `CHANGELOG.md`
- `CITATION.cff`
- `CONTRIBUTING.md`
- `LICENSE`
- `README.md`
- `codemeta.json`
- `pyproject.toml`
- `requirements.txt`
- `setup.py`

### CI and Repository Configuration (3)

- `.flake8`
- `.github/workflows/ci.yml`
- `.gitignore`

### Prior Audit Artifact (1)

- `VERIFY_REORG_REPORT.md`

## Package and Entry-Point Map

`pyproject.toml` is the active package metadata source. `setup.py` delegates to
setuptools with `setup()`, and `requirements.txt` duplicates the runtime
dependency set.

Declared console entry points:

| Command | Target | Status |
| --- | --- | --- |
| `cpd` | `toolkit.cpd_cli:main` | Target exists and imports |
| `cpd-cli` | `toolkit.cpd_cli:main` | Target exists and imports |
| `bocpd-cli` | `changepoint_lab.cli.bocpd_cli:main` | Target exists and imports |
| `within-period-cli` | `changepoint_lab.algorithms.bayesian.within_period.cli:main` | Target exists and imports |

Top-level stable exports from `changepoint_lab.__all__`:

- Version and result types: `__version__`, `ChangePointResult`,
  `SegmentationResult`, `OnlineProbabilityResult`, `PosteriorSampleResult`,
  `LatentStateResult`, `ModelSelectionResult`
- PELT: `PELT`, `pelt`, `gram_rbf`, `kcp_penalized`, `kcp_select_bic`
- BOCPD: `BOCPD`, `BOCPDConfig`, `BOCPDResult`, `Hazard`, `ConstantHazard`, `BoostedBoundaryHazard`, `ScheduledHazard`
- Within-period: `WithinPeriodCPD`
- E-Divisive: `edivisive`, `EDivisive`, `EDivisiveResult`, `EDivisiveSplit`
- State-space: `HSMM`, `HSMMConfig`, `HSMMParams`, `PoissonDur`, `SDHMM`, `SDHMMConfig`, `SDHMMResult`, `SDHMMMixVI`, `SDHMMMixVIConfig`, `SDHMMMixVIResult`
- Kernel: `KernelCPD`
- Deprecated aliases exposed by the compatibility layer: `pelt`, `hsmm`, `sdhmm`, `sdhmm_mix_vi`, `within_period`

Observed compatibility issue: `changepoint_lab.bocpd` is documented in
`paper.md` but does not resolve through the current compatibility layer.

## Algorithm Census

| Method | Primary modules | Public entry points | Source/specification status | Inputs and outputs | Index semantics | Randomness | Tests/examples | Known limitations from audit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PELT | `algorithms/optimization/pelt.py`, `cost_functions.py` | `PELT`, `pelt`, `pelt_detect` | Killick et al. (2012) is the expected primary source; current code needs explicit traceability docs | 1-D numeric sequence plus `SegmentCost`; returns `PELTResult` or `ChangePointResult` | Right-exclusive interior boundaries in `[1, n-1]` | Deterministic | Unit parity, result fields, interoperability, examples | Penalty semantics and cost-specific oracle coverage still need proof |
| BOCPD | `algorithms/bayesian/bocpd/*` | `BOCPD`, `BOCPDConfig`, `BOCPDAlertConfig`, hazards, plotting helpers, `bocpd-cli` | Adams & MacKay (2007) adaptation with Beta-Bernoulli oracle tests | Binary stream for implemented likelihood; wrapper returns explicit alert-policy result | Canonical `P(r_t = 0 | x_1:t)` posterior by default; alert indices come from `BOCPDAlertConfig` | Deterministic for current Beta-Bernoulli path | Hand-computed posterior, independent recursion, normalization, alert policy, approximation diagnostics, edge cases, parity, CLI, docs | PoissonGamma/GaussianNIW contain placeholders; deprecated `cp_scale` compatibility mode is not calibrated |
| Within-period RJMCMC | `algorithms/bayesian/within_period/*`, `scripts/run_within_period_reproduction.py` | `WithinPeriodCPD`, `WithinPeriodCore`, `RJConfig`, `ModelPrior`, reproduction helpers, CLI | Taylor et al. within-period method is partially verified by detailed-balance tests and synthetic reproduction artifacts | Boolean/event sequence over repeated period; returns posterior samples and modal circular changepoints; reproduction script writes JSON/CSV/SVG artifacts | Explicit `periodic_bin_end` circular bin indices modulo `N` | Uses owned `numpy.random.Generator`; parallel tempering uses spawned cold/hot/swap streams | Interoperability, CLI, reproducibility, RJMCMC math, and reproduction workflow tests | Proprietary paper case-study data are not bundled; reproduction workflow uses documented synthetic analogues |
| Sliced Poisson process | `algorithms/point_process/sliced_poisson.py` | `SlicedPoissonCPD`, `SlicedPoissonConfig`, `EventPeriod`, `fit_marked_sliced_poisson` | Martínez-Hernández and Killick (2024) adaptation with analytical and simulated tests | Repeated period event-time sequences; returns typed `SlicedPoissonResult` with generic `SegmentationResult` view | Right-exclusive period-index changepoints | Deterministic optimizer; simulators use owned `numpy.random.Generator` streams | Analytical constant-intensity, exposure, amplitude-recovery, shape-recovery, marked-extension, and public API tests | No parity with private Howz data or supplementary code; shared-baseline marked process raises `NotImplementedError` |
| E-Divisive | `algorithms/nonparametric/edivisive.py`, `edivisive_core.py` | `EDivisive`, `edivisive` | Matteson & James (2014) expected | Numeric sequence/matrix; returns recursive split result and labels | Right-exclusive interior split indices | Uses owned `numpy.random.Generator` for permutation/bootstrap paths and result provenance | Memmap, parity, interoperability, examples, reproducibility tests | Needs independent energy-statistic and permutation oracle validation |
| Kernel CPD / exact KCP | `algorithms/kernel/kcp.py`, `kcp_core.py`, `bandwidth_cv.py` | `KernelCPD`, `gram_rbf`, `kcp_penalized`, `kcp_select_bic` | Kernel changepoint references need registry entry | Numeric matrix or precomputed prefix depending on layer; returns `KCPResult`/`SegmentationResult` | Right-exclusive interior split indices | Mostly deterministic | Segment-bounds, KCP package tests | Broader exact/RFF parity and penalty-selection validation remain pending |
| RFF KCP | `algorithms/kernel/kcp_rff.py`, `rff_variants.py` | Lower-level RFF helpers and CLI adapters | RFF approximation should cite Rahimi & Recht plus KCP source | Numeric matrix; approximate kernel segmentation | Right-exclusive interior split indices | Uses explicit local `numpy.random.Generator` streams | KCP package tests, performance tests | Exact/RFF parity and approximation quality need oracle tests |
| HSMM | `algorithms/state_space/hsmm.py`, `hsmm_core.py`, emissions | `HSMM`, `HSMMConfig`, `HSMMParams`, `PoissonDur`, `NegBinDur` | Yu (2010) expected for HSMM inference | Emission log-likelihood matrix; returns `LatentStateResult` with decoded states, duration-end markers, and changepoints | Wrapper uses nonzero decoded duration-end markers | Uses owned `numpy.random.Generator` and result provenance | Duration cache, emission model, parity tests | Broader duration semantics still need independent validation |
| SD-HMM | `algorithms/state_space/sdhmm.py` | `SDHMM`, `SDHMMConfig`, `SDHMMResult` | Code comment cites Manouchehri & Bouguila (Sensors 2023); needs formal registry | Compositional/proportional matrix; returns state changes | State transition indices from Viterbi diff | Uses owned `numpy.random.Generator` and result provenance | Numerical stability, examples, unit tests | Scaled-Dirichlet gradients and defaults need independent validation |
| SD-HMM Mix VI | `algorithms/state_space/sdhmm_mix_vi.py` | `SDHMMMixVI`, config/result | Needs formal source and derivation record | Compositional matrix; returns state/component metadata | State transition indices from Viterbi diff | Uses owned `numpy.random.Generator` and result provenance | Some tests indirectly; example import fails | Packaged example fails at import; VI math/oracles absent |
| Emission helpers | `algorithms/state_space/emissions/*` | Gaussian diagonal/full and AR helpers | Standard Gaussian/AR specifications implied | Labels/responsibilities or model inputs; returns parameters/log-likelihoods | Per-observation state labels | Seeded helpers use local `numpy.random.Generator` streams | `tests/unit/test_hsmm/test_emission_models.py` | Full oracle coverage still incomplete |
| Utilities and toolkit adapters | `common/*`, `toolkit/api_harmonizer.py`, `toolkit/cpd_cli.py` | Utility functions, CLI, registry adapters | Internal support layer; no scientific source required except statistical helpers | Mixed | Mixed | Sampling helpers use local `numpy.random.Generator` streams | CLI and plotting smoke tests | `toolkit` public status is unclear; adapters use dictionary-heavy results |

## Validation Notes

Commands run during this audit:

- `git ls-files` and `rg --files` for independent file enumeration.
- AST/TOML introspection for modules, definitions, imports, and entry points.
- `pkgutil.walk_packages(changepoint_lab.__path__)` found 64 importable package modules.
- Subprocess import sweep: all core modules import; `changepoint_lab.examples.edivisive_example`, `hsmm_example`, `kcp_example`, and `kcp_rff_example` timed out on import; `changepoint_lab.examples.sdhmm_mix_vi_example` failed on import with `TypeError: 'tuple' object does not support item assignment`.
- CLI help checks passed for `toolkit.cpd_cli`, `changepoint_lab.cli.bocpd_cli`, and `changepoint_lab.algorithms.bayesian.within_period.cli`.
- Editable install in a clean virtual environment succeeded and imported `changepoint_lab.__version__ == "0.1.11"`.
- `python -m build` succeeded and produced wheel and sdist.
- Wheel installation in a clean virtual environment succeeded and imported `changepoint_lab.__version__ == "0.1.11"`.

Minimal public smoke calls:

| Surface | Result |
| --- | --- |
| `PELT(...).fit_predict(...)` | Passed |
| legacy `pelt(...)` | Passed with deprecation warning |
| `EDivisive(...).fit_predict(...)` | Passed |
| `edivisive(...)` | Passed |
| `BOCPD(...).fit_predict(...)` | Passed |
| `WithinPeriodCPD(...).fit_predict(...)` | Failed in the audit harness because `ModelPrior` does not accept `min_seg_len`; the actual field is `l` |
| `KernelCPD(penalty=1.0).fit_predict(...)` | Passed wrapper execution; KCP/RFF now validate interior right-exclusive boundaries |
| legacy `changepoint_lab.bocpd` | Failed with `AttributeError` |

## Empty or Marker Files

The following tracked files are empty. All are package markers or type markers,
not obsolete files by content alone:

- `changepoint_lab/py.typed`
- `changepoint_lab/algorithms/optimization/__init__.py`
- `changepoint_lab/cli/__init__.py`
- `changepoint_lab/common/__init__.py`
- `changepoint_lab/common/diagnostics/__init__.py`
- `changepoint_lab/common/io/__init__.py`
- `changepoint_lab/common/types/__init__.py`
- `changepoint_lab/common/utils/__init__.py`
- `changepoint_lab/core/__init__.py`
- `changepoint_lab/examples/__init__.py`
- `changepoint_lab/tests/test_kcp/__init__.py`
- `examples/integration/__init__.py`
- `tests/unit/test_hsmm/__init__.py`
- `toolkit/__init__.py`
- `toolkit/tests/__init__.py`

## Metadata Reconciliation

- Canonical repository/brand name: `ChangePointLab`
- Canonical import package: `changepoint_lab`
- Canonical distribution name in `pyproject.toml`: `changepoint-lab`
- Historical names to treat as legacy or stale: `cp-ss-toolkit`, `changepoint-toolkit`, `changepoint_toolkit`
- Current version in package metadata, runtime, CFF, and Zenodo JSON: `0.1.11`

No tracked generated build artifacts were present before validation. Build
commands created local `build/`, `dist/`, and `changepoint_lab.egg-info/`
artifacts; these should remain untracked and be removed after validation.
