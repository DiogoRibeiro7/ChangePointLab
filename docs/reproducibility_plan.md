# Reproducibility Roadmap

Date: 2026-07-23

This document is a roadmap, not a description of currently implemented
benchmark, publication, Docker, or Zenodo automation APIs. Earlier aspirational
examples are preserved in `docs/science/claim_audit.md` and should not be cited
as available functionality.

## Current Baseline

- Package metadata, runtime version, `CITATION.cff`, and `.zenodo.json` are
  aligned for the current release series.
- CI runs tests, quality checks, documentation builds, and package builds.
- `docs/science/method_registry.yml` records method sources, code paths, tests,
  deviations, MySense relevance, and verification status.
- `docs/science/claim_audit.md` records unsupported or obsolete scientific and
  documentation claims before they are rewritten.

## Required Before Reproducibility Claims

1. Add independent oracle tests for each method before marking it `verified`.
2. Add deterministic fixture datasets with licenses and checksums.
3. Record benchmark environment metadata and random seeds.
4. Generate benchmark tables and figures from committed scripts.
5. Run documentation examples from an installed wheel.
6. Archive only coherent release states through GitHub and Zenodo.

## Out of Scope

- PyPI publishing.
- JOSS submission.
- Claims of paper parity, statistical calibration, accuracy superiority, runtime
  superiority, or broad reproducibility without generated evidence.
