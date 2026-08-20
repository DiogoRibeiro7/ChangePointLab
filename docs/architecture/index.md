# Architecture

ChangePointLab uses a `src/` layout with one import package,
`changepoint_lab`. The package currently keeps production runtime code under
`src/changepoint_lab/`, tests under `tests/`, and scientific traceability under
`docs/science/`.

The sections below distinguish the current checked-in layout from desired
future structure. Paths in the current section are expected to exist in the
repository and are covered by a drift test.

## Current Source Layout

<!-- architecture-current-start -->

- `src/changepoint_lab/`
  - `src/changepoint_lab/__init__.py` defines the stable top-level facade.
  - `src/changepoint_lab/api_status.py` records lifecycle status for top-level
    exports.
  - `src/changepoint_lab/_compat.py` contains lazy deprecated aliases.
  - `src/changepoint_lab/_optional.py` centralizes optional dependency loading.
  - `src/changepoint_lab/py.typed` marks the package as typed.

- `src/changepoint_lab/algorithms/`
  - `src/changepoint_lab/algorithms/_base.py` holds shared detector base
    helpers.
  - `src/changepoint_lab/algorithms/optimization/pelt.py` implements PELT and
    optimal partitioning utilities.
  - `src/changepoint_lab/algorithms/optimization/cost_functions.py` contains
    segment costs and penalty helpers used by optimization methods.
  - `src/changepoint_lab/algorithms/nonparametric/edivisive.py` exposes the
    estimator wrapper.
  - `src/changepoint_lab/algorithms/nonparametric/edivisive_core.py` contains
    the E-Divisive implementation.
  - `src/changepoint_lab/algorithms/kernel/kcp.py` exposes the Kernel CPD
    estimator wrapper.
  - `src/changepoint_lab/algorithms/kernel/kcp_core.py` contains exact kernel
    segmentation routines.
  - `src/changepoint_lab/algorithms/kernel/kcp_rff.py` contains random Fourier
    feature approximations.
  - `src/changepoint_lab/algorithms/kernel/bandwidth_cv.py` contains bandwidth
    selection utilities.
  - `src/changepoint_lab/algorithms/kernel/rff_variants.py` contains additional
    RFF maps.
  - `src/changepoint_lab/algorithms/point_process/sliced_poisson.py` implements
    the sliced Poisson detector and result types.

- `src/changepoint_lab/algorithms/bayesian/`
  - `src/changepoint_lab/algorithms/bayesian/bocpd/` is a package, not a single
    module.
  - `src/changepoint_lab/algorithms/bayesian/bocpd/__init__.py` re-exports the
    BOCPD public package surface.
  - `src/changepoint_lab/algorithms/bayesian/bocpd/core.py` contains BOCPD
    state updates, hazards, configuration, results, and alert extraction.
  - `src/changepoint_lab/algorithms/bayesian/bocpd/likelihoods.py` contains
    conjugate likelihood implementations.
  - `src/changepoint_lab/algorithms/bayesian/bocpd/validation.py` contains
    BOCPD-specific validation helpers.
  - `src/changepoint_lab/algorithms/bayesian/bocpd/plotting.py` contains
    optional plotting helpers.
  - `src/changepoint_lab/algorithms/bayesian/within_period/within_period_cpd.py`
    contains the within-period detector.
  - `src/changepoint_lab/algorithms/bayesian/within_period/anchor_utils.py`
    contains circular anchor helpers.
  - `src/changepoint_lab/algorithms/bayesian/within_period/posterior_predictive.py`
    contains posterior predictive helpers.
  - `src/changepoint_lab/algorithms/bayesian/within_period/replication.py`
    contains reproduction helpers.
  - `src/changepoint_lab/algorithms/bayesian/within_period/cli.py` contains the
    within-period command-line entry point.
  - `src/changepoint_lab/algorithms/bayesian/within_period/samplers/tempering.py`
    contains parallel-tempering support.

- `src/changepoint_lab/algorithms/state_space/`
  - `src/changepoint_lab/algorithms/state_space/hsmm.py` exposes the HSMM
    estimator wrapper.
  - `src/changepoint_lab/algorithms/state_space/hsmm_core.py` contains the core
    HSMM dynamic programs.
  - `src/changepoint_lab/algorithms/state_space/sdhmm.py` contains experimental
    SD-HMM support.
  - `src/changepoint_lab/algorithms/state_space/sdhmm_mix_vi.py` contains
    experimental mixture-VI SD-HMM support.
  - `src/changepoint_lab/algorithms/state_space/emissions/gaussian_diag.py`
    contains diagonal Gaussian emissions.
  - `src/changepoint_lab/algorithms/state_space/emissions/gaussian_full.py`
    contains full-covariance Gaussian emissions.
  - `src/changepoint_lab/algorithms/state_space/emissions/ar_emissions.py`
    contains autoregressive emissions.

- `src/changepoint_lab/core/`
  - `src/changepoint_lab/core/datatypes.py` defines result dataclasses,
    protocols, and boundary-convention metadata.
  - `src/changepoint_lab/core/segmentation.py` contains boundary conversion and
    segment-label helpers.
  - `src/changepoint_lab/core/random.py` contains deterministic RNG ownership
    helpers.
  - `src/changepoint_lab/core/validation.py` contains shared public input
    validation.

- `src/changepoint_lab/common/`
  - `src/changepoint_lab/common/api_harmonizer.py` contains legacy API
    harmonization helpers.
  - `src/changepoint_lab/common/diagnostics/diagnostics.py` contains diagnostic
    calculations.
  - `src/changepoint_lab/common/io/data_loader.py` and
    `src/changepoint_lab/common/io/io_utils.py` contain data loading and result
    serialization helpers.
  - `src/changepoint_lab/common/plotting/plotting_helpers.py`,
    `src/changepoint_lab/common/plotting/edivisive_plotting.py`, and
    `src/changepoint_lab/common/plotting/kcp_plotting.py` contain optional
    plotting helpers.
  - `src/changepoint_lab/common/types/types.py` and
    `src/changepoint_lab/common/utils/utils.py` contain general utility types
    and helpers.

- `src/changepoint_lab/cli/`
  - `src/changepoint_lab/cli/cpd_cli.py` contains the broader command-line
    wrapper.
  - `src/changepoint_lab/cli/bocpd_cli.py` contains BOCPD-specific command-line
    support.

<!-- architecture-current-end -->

## Public Surface

The stable top-level API is re-exported from `changepoint_lab`. Lifecycle status
is explicit in `changepoint_lab.api_status.API_MANIFEST`, with convenience
groups exposed as `changepoint_lab.__stable__`,
`changepoint_lab.__experimental__`, and `changepoint_lab.__deprecated__`.

Boundary conventions are represented in result objects from
`src/changepoint_lab/core/datatypes.py` and conversion helpers in
`src/changepoint_lab/core/segmentation.py`. Stochastic code should use owned
`numpy.random.Generator` streams created or derived through
`src/changepoint_lab/core/random.py`.

## Target Architecture

The following modules remain possible future structure. They are not part of the
current package tree and should not be described as present until implemented:

- A detector registry for name-to-class discovery and CLI composition.
- A core metrics module for changepoint precision, recall, matching, and
  Hausdorff-style distances.
- Dataset packages for synthetic and license-safe real-world examples.
- A unified plotting facade that sits above current method-specific plotting
  helpers.
- Benchmark harness modules for reproducible comparisons across detectors.
- Versioned serialization wrappers for all result objects.

## Rationale

- A single `algorithms/` parent keeps detector implementations discoverable.
- `core/` hosts shared contracts: result objects, boundary semantics,
  validation, and RNG ownership.
- `common/` contains older general-purpose utilities and optional plotting or
  I/O helpers. New stable cross-cutting contracts should prefer `core/` when the
  behavior is part of the public runtime surface.
- BOCPD is organized as a package because the implementation is split across
  core state updates, likelihoods, validation, and optional plotting.
- The `src/` layout prevents accidental imports from the repository root during
  local development.
