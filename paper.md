# Scholarly Claim Status

Date: 2026-07-23

This repository is not preparing a JOSS submission. The active scholarly record
for the current release work is the Zenodo archive metadata, `CITATION.cff`, and
the traceability documents under `docs/science/`.

The previous manuscript-style text made broad claims about implementation
coverage, optimization, applications, and benchmark performance. Those claims
are preserved and classified in `docs/science/claim_audit.md` before further
rewrites. Scientific method traceability is maintained in
`docs/science/method_registry.yml` and rendered in
`docs/science/method_registry.md`.

Before any manuscript or publication text is restored, each included method must
have:

- a registry entry with citation, code paths, tests, deviations, and verification
  status;
- executable examples using the installed package;
- independent oracle tests or measured artifacts for any parity, performance, or
  accuracy claim;
- release metadata aligned with the Zenodo archive.
