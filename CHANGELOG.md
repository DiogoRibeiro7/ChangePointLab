# Changelog

## [Unreleased]
### Fixed
- Canonicalized sliced Poisson exposure intervals so overlapping, nested,
  duplicate, and touching windows count as a half-open observed-time union.
- Replaced sliced Poisson global midpoint exposure masking with interval-local
  quadrature so narrow observed windows contribute positive exposure.
- Tracked sliced Poisson event totals with direct integer prefixes instead of
  reconstructing counts from floating B-spline sufficient statistics.
- Made sliced Poisson segment optimizer failures raise by default or become
  infinite-cost fits under explicit non-raising policies.

## [0.1.15] - 2026-08-20
### Added
- Added baseline audit artifacts and tests that pin current behavior before
  follow-up corrections.
- Added stable API type-check coverage and documented typing support.
- Added result invariant tests and documentation for boundary conventions,
  serialized payloads, and result validation.
- Added centralized public input validation helpers and coverage.
- Added machine-readable API lifecycle metadata, public API status docs, and
  migration guidance.
- Added an architecture documentation drift test for documented current paths.
- Added independent E-Divisive energy-statistic, resampling, null-calibration,
  and power-curve validation tests.
- Exposed E-Divisive significance, recursion, resampling, chunking, memmap,
  progress, and sequential execution controls through the wrapper and CLI.

### Changed
- Expanded Ruff, mypy, and CI quality gates across stable package surfaces.
- Reconciled the audit roadmap and risk register with the current dependency
  order.
- Separated stable, experimental, and deprecated top-level exports.
- Rewrote architecture documentation to distinguish current package layout from
  possible future structure.

### Fixed
- Hardened result objects against invalid boundaries, labels, metadata, and
  non-finite score values.
- Normalized deprecated compatibility warnings with documented replacement paths
  and removal versions.
- Fixed NumPy 1.21 runtime compatibility for eager array type aliases.
- Corrected E-Divisive split search so the last admissible split
  `m - min_size` is considered.

## [0.1.14] - 2026-07-23
### Added
- Added exact/RFF KernelCPD correctness oracles, dense Gram validation, and
  wrapper metadata for kernel/RFF approximation controls.

### Fixed
- Fixed KCP bandwidth cross-validation changepoint scoring so it uses
  `KCPResult.total_cost` instead of falling back through invalid scores.

## [0.1.13] - 2026-07-23
### Fixed
- Corrected PELT candidate handling so bundled costs match exhaustive
  optimal-partitioning oracles under minimum segment constraints.

### Changed
- Documented PELT penalty units, exact candidate-retention behavior, and
  concave-penalty approximation limits.

## [0.1.12] - 2026-07-23
### Added
- Added scalar Poisson-Gamma BOCPD likelihood support, explicit likelihood
  injection, batch streaming updates, missing-observation handling, and
  checkpoint/resume state dictionaries.

## [0.1.11] - 2026-07-23
### Changed
- Made BOCPD use the canonical unscaled run-length posterior by default,
  deprecated `cp_scale`, and moved wrapper changepoint alerts to an explicit
  `BOCPDAlertConfig` post-processing policy with evidence and approximation
  diagnostics.

## [0.1.10] - 2026-07-23
### Added
- Added a dedicated sliced Poisson process detector for repeated event-time
  periods, including B-spline IHPP segment costs, PELT optimization, exposure
  intervals, independent marked-process fitting, simulations, and diagnostics.

## [0.1.9] - 2026-07-23
### Added
- Added a within-period reproduction workflow with paper-style synthetic
  scenarios, a MySense-style generated sensor example, CSV/JSON/SVG artifact
  output, and notebook execution coverage.

## [0.1.8] - 2026-07-23
### Added
- Added within-period RJMCMC diagnostics for acceptance rate and proposal move
  counts.
- Added tiny-state within-period RJMCMC tests for detailed balance, empirical
  stationary frequencies, `pois_lambda`, and rotation invariance.

### Fixed
- Corrected the within-period circular state representation, truncated Poisson
  segment-count prior, and exact reverse proposal probability accounting.

## [0.1.7] - 2026-07-23
### Added
- Added stochastic result provenance and reproducibility guidance.
- Added tests for seeded replay, spawned stream independence, and global RNG
  isolation.

### Changed
- Migrated stochastic production paths to owned `numpy.random.Generator`
  streams instead of module-level NumPy or Python random state.
- Updated within-period baseline traces to the explicit-Generator behavior.

## [0.1.6] - 2026-07-23
### Added
- Added canonical linear and circular changepoint conversion helpers.
- Documented right-exclusive linear boundaries and periodic bin-end circular
  boundaries.

### Changed
- Marked within-period wrapper results with explicit `periodic_bin_end`
  boundary semantics.

### Fixed
- Fixed exact and RFF KCP backtracking so terminal endpoint `n` is not emitted
  as a changepoint.

## [0.1.5] - 2026-07-23
### Added
- Added typed public result contracts for segmentation, online probability,
  posterior-sampling, latent-state, and model-selection outputs.
- Added behavior tests for all stable top-level estimator exports.

### Fixed
- Fixed `KernelCPD.fit_predict` so the wrapper builds a kernel prefix before
  calling the core KCP routine.
- Fixed HSMM wrapper changepoints by extracting nonzero duration-end markers.
- Fixed `SDHMMMixVI.fit_predict` parameter updates that assigned into immutable
  tuples.

## [0.1.4] - 2026-07-23
### Changed
- Reduced the core runtime dependency set to NumPy and moved Matplotlib and
  pandas behind optional extras with lazy import errors.
- Added Python 3.10 through 3.14 compatibility policy and CI coverage for
  minimum and newest core dependency combinations.

### Fixed
- Recorded the Python 3.14 baseline exception text for the known within-period
  tiny-input failure path.

## [0.1.3] - 2026-07-23
### Added
- Added scientific method traceability and claim-audit documentation.
- Added deterministic baseline fixtures and characterization tests for public
  wrappers and low-level algorithm entry points.

### Changed
- Replaced unsupported benchmark, citation, and publication-scope claims with
  evidence-first status notes.
- Migrated packaging to Poetry with a `src/changepoint_lab` layout and
  package-local CLI entry points.

### Removed
- Removed duplicate `setup.py` and `requirements.txt` packaging paths.
- Removed the obsolete `toolkit` package from the distribution.

## [0.1.1] - 2026-07-23
### Fixed
- Declared CLI runtime dependencies required by the installed entry points.

## [0.1.0] - 2026-07-23
### Added
- Introduced `changepoint_lab` package directory with unified `algorithms/` tree and `py.typed` marker.

### Deprecated
- Legacy imports under `changepointlab.*` and top-level modules such as `pelt`, `bocpd`, `edivisive`, `hsmm`, and `kcp` now emit `DeprecationWarning`.
- These shims will be removed in **v0.11.0**; attribute fallbacks in `_compat` will be removed in **v0.12.0**.

### Migration
- Old: `from changepointlab.optimization import pelt`
- New: `from changepoint_lab import PELT`

