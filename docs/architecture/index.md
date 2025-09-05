# Architecture

ChangePointLab groups code into a single `changepoint_lab/` package with clear
responsibilities for each subfolder. The layout below shows the main
directories and their roles.

```
changepoint_lab/
├── algorithms/                        # All algorithms live here (single parent)
│   ├── _base.py                       # Abstract interfaces, shared mixins for algorithms
│   ├── bayesian/
│   │   ├── __init__.py
│   │   ├── bocpd.py                   # Bayesian Online CPD
│   │   └── within_period.py           # Within-period Bayesian detector (if present)
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── pelt.py                    # PELT / optimal partitioning variants
│   │   └── cost_functions.py          # Cost / penalty functions specific to optimization methods
│   ├── nonparametric/
│   │   ├── __init__.py
│   │   ├── edivisive.py               # E-Divisive wrapper
│   │   └── edivisive_core.py          # Core E-Divisive implementation
│   ├── state_space/
│   │   ├── __init__.py
│   │   ├── hsmm.py                    # HSMM wrapper
│   │   ├── hsmm_core.py               # Core HSMM implementation
│   │   └── emissions/                 # Emission model helpers
│   └── kernel/
│       ├── __init__.py
│       ├── kcp.py                     # Kernel CPD wrapper
│       ├── kcp_core.py                # Core KCP implementation
│       ├── kcp_rff.py                 # RFF utilities
│       ├── bandwidth_cv.py            # Bandwidth selection
│       └── rff_variants.py            # Advanced RFF variants
│
├── core/                              # Cross-cutting “engine” pieces shared by multiple algos
│   ├── __init__.py
│   ├── datatypes.py                   # Typed result objects, aliases (e.g., NDArray), Protocols
│   ├── exceptions.py                  # Library-specific exceptions
│   ├── preprocessing.py               # Standardizers, segment helpers, windowing
│   ├── metrics.py                     # F1@tol, Hausdorff-like, precision/recall@k, etc.
│   ├── validation.py                  # Input validation utilities (shapes, dtypes)
│   └── registry.py                    # Optional: name→class registry for discovery/CLI
│
├── common/                            # General-purpose utilities (no algorithm knowledge)
│   ├── __init__.py
│   ├── io.py                          # Save/load results; simple file I/O
│   ├── utils.py                       # Logging setup, random seeds, misc helpers
│   └── typing.py                      # Public typing utilities & Literal choices (if needed)
│
├── datasets/
│   ├── __init__.py
│   ├── synthetic.py                   # Generators for reproducible synthetic benchmarks
│   └── real_world.py                  # (Optional) loaders with license-safe examples
│
├── visualization/                     # User-facing plotting APIs
│   ├── __init__.py
│   └── plotting.py                    # High-level plots (signals, change points, diagnostics)
│
├── benchmarks/
│   ├── __init__.py
│   └── comparison.py                  # Reproducible comparisons across algorithms
│
├── tests/                             # (Repository-level tests; not installed)
│   └── ...                            # Unit/integration tests mirroring new structure
│
├── __init__.py                        # Clean public API; re-exports major classes/functions
├── _compat.py                         # Compatibility shims for old import paths
└── py.typed                           # PEP 561 type marker
```

## Rationale

- **Single parent `algorithms/`** centralises all detectors for straightforward discovery.
- **`core/`** hosts shared types and mechanics, reducing duplication and avoiding circular imports.
- **`common/` vs `visualization/`** keeps low-level helpers separate from user-facing plotting.
- Shipping a `py.typed` marker advertises type information to downstream users.
