# Release checklist: v0.1.17

This is a release candidate, not evidence that the GitHub release or Zenodo
archive already exists. The latest *published* release remains v0.1.16 until
the steps below are complete.

## Scope

- Experimental changepoint evaluation and offline detector comparison APIs.
- Reproducible Gaussian Monte Carlo benchmarking, with no asserted detector
  ranking or measured performance advantage.
- Synthetic stress-test scenarios for within-segment AR(1) autocorrelation,
  additive Gaussian outliers, and continuous linear drift.

## Before merging the release-preparation PR

1. Confirm every required GitHub Actions check passes on the candidate commit,
   including supported Python versions, cross-platform checks, documentation,
   vulnerability and license audits, and distribution validation.
2. Check `pyproject.toml`, runtime `__version__`, Sphinx `release`,
   `CITATION.cff`, and `.zenodo.json` all identify `0.1.17`.
3. Confirm the release date in `CITATION.cff`, `.zenodo.json`, and `CHANGELOG.md`
   matches the **actual** intended release date. Update it if publication occurs
   after 2026-11-16.
4. Verify the wheel and source archive contain the new public APIs and
   `py.typed`, with no test fixture or secret files.
5. Review documentation and claims: experimental benchmarking features do not
   imply an externally validated detector ranking.

## Publishing (after the PR is merged)

1. Create an annotated `v0.1.17` tag pointing to the validated commit on `main`.
2. Publish the matching GitHub Release with the changelog summary.
3. Confirm the connected Zenodo integration ingests that release and generates
   the version-specific DOI. Investigate a failed ingestion before proceeding.
4. Verify the Zenodo record title, authors, version, publication date, license,
   archived files, and related identifiers against the GitHub release.
5. Keep the existing concept DOI for generic project references; use the new
   version-specific DOI when citing reproducible v0.1.17 experiments.
6. Only then update the roadmap's "latest verified release" marker in a
   follow-up PR.

No tag, GitHub Release, or Zenodo deposit is created by this checklist.
