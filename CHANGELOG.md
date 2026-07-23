# Changelog

## [Unreleased]

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

