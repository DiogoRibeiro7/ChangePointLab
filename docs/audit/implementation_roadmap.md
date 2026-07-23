# Implementation Roadmap

Date: 2026-07-23
Scope: ordering plan after repository forensic audit

The roadmap orders scientific correctness and behavioral traceability before API
cleanup, optimization, documentation polish, and release expansion.

## Phase 0: Guardrails Already in Place

- Repository is public and default branch is `main`.
- Latest verified release is `v0.1.6`.
- CI builds docs and distributions and runs tests on supported Python versions.
- Local workflow notes remain untracked.

## Phase 1: Traceability and Baseline

1. Create a method registry mapping each public method to citations, equations,
   code paths, tests, examples, deviations, and verification status.
2. Audit README, docs, paper, metadata, and examples for unsupported claims.
3. Freeze behavioral baselines for all exported estimators and low-level
   algorithm entry points.
4. Add small analytical or brute-force expected results for methods where
   correctness fixes are planned.

Exit criteria:

- Every stable public method has a primary source or explicit experimental label.
- Every current public wrapper is either exercised or documented as broken.
- Planned behavior changes have pre-fix characterization cases.

## Phase 2: Packaging and API Contract

1. Decide whether to keep setuptools or migrate to Poetry and `src/` layout.
2. Make `pyproject.toml` the only dependency source and retire
   `requirements.txt`.
3. Define public result contracts with typed domain objects, not primary
   `dict[str, Any]` outputs.
4. Separate stable exports from compatibility aliases and migration shims.
5. Fix import safety for packaged examples or remove examples from package
   discovery.

Exit criteria:

- Installed wheel has no broken importable modules.
- All CLI entry points exercise stable public APIs.
- Compatibility paths are finite, tested, and documented.

## Phase 3: Cross-Algorithm Semantics

1. Standardize boundary/index conventions across all algorithms.
2. Document half-open segment semantics and circular within-period semantics.
3. Make randomness explicit with `numpy.random.Generator` or owned RNG state.
4. Add reproducibility tests that run in fresh Python processes.

Exit criteria:

- Every algorithm documents input shape, output indices, segment edges, and RNG behavior.
- No stable public path mutates global RNG state.

## Phase 4: Scientific Corrections and Oracles

Address high-risk methods in dependency order:

1. Within-period RJMCMC posterior, Poisson prior, proposal probabilities, and
   detailed-balance tests.
2. BOCPD canonical probabilities, likelihood support, and extraction policy.
3. PELT cost and penalty oracle tests.
4. KernelCPD wrapper repair and exact/RFF parity.
5. E-Divisive statistic and resampling validation.
6. HSMM Viterbi duration semantics and duration learning validation.
7. SD-HMM and emission helper validation.
8. Missing sliced Poisson process implementation, if retained as a project goal.

Exit criteria:

- Each scientific behavior change has a characterization test, independent
  expected result, regression test, and migration note.
- No paper-parity or performance claim is made without reproducible evidence.

## Phase 5: Reliability, Benchmarks, and Documentation

1. Add numerical edge-case tests across empty, tiny, constant, NaN/Inf, extreme,
   and degenerate inputs.
2. Add property/metamorphic tests for invariants that do not require exact
   changepoint locations.
3. Build benchmark artifacts with environment and commit metadata.
4. Convert quickstarts and tutorials to executable examples.
5. Rewrite comparison and scholarly documents around measured evidence only.

Exit criteria:

- Documentation examples run from the installed package.
- Benchmark claims are generated from versioned artifacts.
- Unsupported or aspirational text is clearly labelled or removed.

## Phase 6: Release Engineering

1. Add vulnerability/license audit commands and optional SBOM generation.
2. Validate wheel/sdist contents before every GitHub/Zenodo release.
3. Keep PyPI publishing out of scope unless explicitly reintroduced.
4. Keep JOSS submission out of scope unless explicitly reintroduced.
5. Create GitHub/Zenodo releases only when the repository state is coherent and
   relevant checks pass.

Exit criteria:

- Release artifacts carry consistent metadata and version provenance.
- Zenodo metadata matches the repository release.
- No critical or high-severity correctness risk remains for any advertised
  stable scientific behavior.

## Immediate Next Work

The next safe step is the scientific traceability pass:

1. Create `docs/science/method_registry.yml`.
2. Render `docs/science/method_registry.md`.
3. Create `docs/science/claim_audit.md`.
4. Add tests that every package-level stable method has a registry entry and
   that registry paths exist.
