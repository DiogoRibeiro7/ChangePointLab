# Benchmark Report Status

Date: 2026-07-23

The previous benchmark report contained placeholder figures, unreleased version
labels, and numeric superiority claims that were not backed by generated
artifacts in this repository. Those claims are preserved in
`docs/science/claim_audit.md` and are no longer active documentation.

## Current Benchmark Policy

- Do not publish numeric accuracy, runtime, memory, or superiority claims unless
  they are generated from versioned benchmark code and committed artifacts.
- Record the package version, commit SHA, operating system, Python version,
  dependency versions, hardware, datasets, random seeds, and timeout policy for
  every benchmark table.
- Keep external datasets optional, licensed, cached, and checksummed.
- Prefer executable scripts over hand-entered tables.

## Current Status

No cross-library benchmark table is currently verified for ChangePointLab
0.1.12. Existing tests exercise package behavior, CLI integration, and selected
edge cases, but they are not a benchmark suite.

Restoring a comparison report requires a dedicated benchmark harness and review
against `docs/science/method_registry.yml`.
