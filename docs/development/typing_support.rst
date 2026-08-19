Typing Support
==============

ChangePointLab ships a ``py.typed`` marker. The supported typing contract is the
stable public API re-exported from ``changepoint_lab`` plus the shared result
objects and estimator protocols in ``changepoint_lab.core``.

Guaranteed Surface
------------------

The following surfaces are checked by mypy:

* top-level exports from ``changepoint_lab``;
* shared result dataclasses in ``changepoint_lab.core.datatypes``;
* core segmentation and randomness helpers;
* stable estimator wrappers for PELT, BOCPD, E-Divisive, KernelCPD,
  SlicedPoissonCPD, HSMM, SDHMM, SDHMMMixVI, and WithinPeriodCPD;
* downstream-style fixtures under ``tests/typecheck``.

Result objects normalize NumPy-compatible inputs at runtime, but their public
attributes are typed as normalized NumPy arrays after construction. Metadata and
provenance mappings intentionally remain ``Mapping[str, Any]`` because they
carry algorithm-specific diagnostics and backwards-compatible extension data.

Dynamic Boundaries
------------------

Compatibility attributes such as legacy module aliases are provided through
``changepoint_lab.__getattr__`` and may be typed as ``Any`` by downstream tools.
They are retained for migration, not as the preferred typed API.

Optional dependencies are imported lazily. Plotting and pandas-backed helpers
therefore expose typed project call sites, while the imported third-party module
objects remain dynamic at the dependency boundary.

Internal numerical kernels, experimental research extensions, and compatibility
shims may have narrower typing guarantees until they become part of the stable
documented API.
