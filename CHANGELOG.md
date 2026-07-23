# Changelog

## [Unreleased]

## [0.1.0] - 2026-07-23
### Added
- Introduced `changepoint_lab` package directory with unified `algorithms/` tree and `py.typed` marker.

### Deprecated
- Legacy imports under `changepointlab.*` and top-level modules such as `pelt`, `bocpd`, `edivisive`, `hsmm`, and `kcp` now emit `DeprecationWarning`.
- These shims will be removed in **v0.11.0**; attribute fallbacks in `_compat` will be removed in **v0.12.0**.

### Migration
- Old: `from changepointlab.optimization import pelt`
- New: `from changepoint_lab import PELT`

