# Implementation Roadmap

Date: 2026-07-23
Scope: ordering plan after repository forensic audit

The roadmap orders scientific correctness and behavioral traceability before API
cleanup, optimization, documentation polish, and release expansion.

Current review update: 2026-08-19 at commit
`741e6cce0edf517dd0c9e4f9a2b562c55f2e5cfe`.

The July phase plan is retained as historical context. The active dependency
sequence is now the "Current Dependency-Aware Roadmap" section below; it reflects
completed traceability work, baseline capture, and newly identified sliced
Poisson correctness blockers.

## Phase 0: Guardrails Already in Place

- Repository is public and default branch is `main`.
- Latest verified release is `v0.1.14`.
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

This section is superseded by the current dependency-aware roadmap. The
scientific traceability files now exist at `docs/science/method_registry.yml`,
`docs/science/method_registry.md`, and `docs/science/claim_audit.md`.

## Current Dependency-Aware Roadmap

### Gate A: foundation and repository truth

Complete the remaining foundation work before scientific behavior changes:

1. Expand mypy coverage for stable public surfaces.
2. Expand Ruff and code-quality gates in measured steps.
3. Strengthen result-object invariants.
4. Centralize public input validation.
5. Clean public API and compatibility paths.
6. Synchronize architecture docs with the current package tree.
7. Choose and enforce one canonical documentation build.
8. Harden CI coverage and security checks.
9. Add property, metamorphic, numerical-stability, and benchmark harness layers.

Exit criteria:

- Baseline failures and skips stay explicit in `docs/audit/baseline_test_truth.md`.
- Static and documentation gates reflect the stable API without unsupported claims.
- No scientific verification status changes without new independent evidence.

### Gate B: sliced Poisson correctness

Repair correctness issues in dependency order:

1. Canonicalize exposure intervals into non-overlapping observed windows.
2. Replace fixed-grid exposure accounting with interval-aware integration.
3. Track direct integer event sufficient statistics.
4. Define optimizer-failure propagation policy for segment costs.

Exit criteria:

- Overlap, nested-window, duplicate-window, and sub-grid exposure cases have
  analytical tests.
- Non-converged segment fits cannot silently affect changepoint selection.
- The `sliced_poisson_process` registry entry remains `partially_verified` until
  external or independent parity evidence exists.

### Gate C: point/count process expansion

After Gate B, add count and marked-process features:

1. Piecewise-constant Poisson PELT cost with exposure.
2. Gamma-Poisson marginal and negative-binomial segment costs.
3. Unified count/exposure data contract.
4. Shared-baseline and joint marked-process segmentation.
5. Continuous-time Poisson segmentation, simulator hardening, power benchmarks,
   plotting, CLI support, and replication scaffolding.

### Gate D: BOCPD likelihoods and state

1. Implement univariate Gaussian Normal-Inverse-Gamma BOCPD.
2. Complete multivariate Gaussian Normal-Inverse-Wishart BOCPD.
3. Version checkpoint schemas before adding more stateful likelihoods.
4. Add adaptive run-length pruning, exposure-aware Poisson-Gamma, count
   overdispersion, hazard validation, and alert calibration.

Exit criteria:

- Gaussian paths are not advertised as supported until independent conjugate
  predictive/update oracles pass.
- Checkpoint migrations are versioned and backward compatible.

### Gate E: PELT and model selection

1. Define the segment-cost protocol and cache contract.
2. Add safe real PELT pruning only after the contract is explicit.
3. Add CROPS, concave-penalty diagnostics, trend costs, and constrained
   candidate boundaries as separate reviewable changes.

### Gate F: kernel, E-Divisive, state-space, and within-period

Proceed independently after Gate A, with oracle evidence required for status
changes:

1. Kernel model-selection calibration, Nyström approximation, RFF error
   contracts, and memory-bounded exact kernels.
2. E-Divisive multiple-testing policy, dependent resampling, and distance-engine
   performance.
3. HSMM oracles, count emissions, learning diagnostics, SD-HMM objective
   validation, and safe serialization.
4. Within-period multichain diagnostics, posterior credible summaries, and
   hierarchical multisensor modeling.

### Gate G: package capabilities

1. Core changepoint evaluation metrics.
2. Synthetic datasets.
3. Detector registry and unified comparison API.
4. Transparent consensus ensembles.
5. Versioned result serialization.
6. Optional pandas adapters, unified plotting, unified CLI, executable examples,
   and traceability automation.

### Gate H: experimental research

Keep online/offline hybrid refinement, renewal-process detectors, Hawkes-process
regime changes, and count-dispersion diagnostics out of the stable API until
their assumptions, validation data, and registry entries are explicit.

### Gate I: external validation and release

Run external oracle validation, benchmark claim gating, adversarial scientific
review, and release planning only when critical and high-severity scientific
blockers are closed with executable evidence.
