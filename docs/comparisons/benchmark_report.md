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
0.1.15. Existing tests exercise package behavior, CLI integration, selected
edge cases, and a local benchmark-harness smoke run, but they are not a
cross-library benchmark suite.

## Versioned Smoke Artifact

The first committed harness artifact is intentionally small:

- Raw JSON: `docs/comparisons/artifacts/smoke_benchmark.json`
- Raw CSV: `docs/comparisons/artifacts/smoke_benchmark.csv`
- Generated Markdown: `docs/comparisons/benchmark_smoke.md`

The smoke profile uses deterministic measurement placeholders so regeneration is
stable in CI. It still records environment metadata, dataset hashes, detector
configuration, boundary accuracy, and exact-vs-approximate kernel error fields.

Regenerate the smoke artifact:

```bash
python scripts/benchmark_harness.py --profile smoke --deterministic-measurements
```

Run a local timing and peak-memory profile:

```bash
python scripts/benchmark_harness.py --profile full
```

Restoring broader comparison claims still requires reviewed full-profile
artifacts and method-registry updates.
