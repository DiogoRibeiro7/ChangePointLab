Input Contracts
===============

Public inputs keep detector-specific statistical domains. Shared validators in
``changepoint_lab.core.validation`` cover common structural checks without
forcing every method into one universal data model.

.. list-table::
   :header-rows: 1

   * - Surface
     - Accepted domain
     - Not accepted
   * - ``BaseDetector`` wrappers
     - Non-empty NumPy arrays with one or two dimensions
     - Python lists, empty arrays, arrays with more than two dimensions
   * - ``PELT``, ``KernelCPD``, ``EDivisive``
     - Numeric one- or two-dimensional observations, with method-specific size checks
     - NaN/Inf in low-level numeric routines
   * - ``BOCPD`` with ``BetaBernoulli``
     - Legacy scalar observations coerced by Python truthiness
     - Missing or non-scalar observations in the online update path
   * - ``BOCPD`` with ``PoissonGamma``
     - Finite non-negative integer counts
     - Bool values, negative counts, fractional counts
   * - ``WithinPeriodCPD``
     - One-dimensional binary activity series
     - Non-binary values and empty series
   * - Time and matrix helpers
     - Strictly increasing finite time vectors; finite square matrices
     - Ties or decreases in time, non-square matrices, non-PSD matrices when PSD is required

Stable public input errors use ``TypeError`` for wrong container type and
``ValueError`` for invalid values, shapes, or domain violations.
