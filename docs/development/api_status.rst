API Status
==========

``changepoint_lab.api_status.API_MANIFEST`` is the machine-readable source for
package-level lifecycle status. Tests assert that every stable and experimental
symbol imports, deprecated aliases warn with a removal version, and experimental
symbols stay out of the stable set.

Lifecycle Table
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 18 40 17

   * - Status
     - Symbols
     - Meaning
     - Migration
   * - Stable
     - ``changepoint_lab.__stable__``
     - Supported package-level API. These symbols also appear in ``__all__``.
     - Prefer these names for new code.
   * - Experimental
     - ``SDHMM``, ``SDHMMConfig``, ``SDHMMResult``, ``SDHMMMixVI``,
       ``SDHMMMixVIConfig``, ``SDHMMMixVIResult``
     - Importable for evaluation, but not stable support. Validation remains
       limited and behavior may change before promotion.
     - Pin versions and treat these as opt-in implementations.
   * - Deprecated
     - ``pelt``, ``hsmm``, ``sdhmm``, ``sdhmm_mix_vi``, ``within_period``
     - Lazy compatibility aliases. Each alias emits ``DeprecationWarning`` and
       records a finite removal version in the manifest.
     - Migrate before ``0.3.0``.

Deprecated Aliases
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Alias
     - Replacement
     - Removal
   * - ``pelt``
     - ``changepoint_lab.algorithms.optimization.pelt.pelt``
     - ``0.3.0``
   * - ``hsmm``
     - ``changepoint_lab.HSMM``
     - ``0.3.0``
   * - ``sdhmm``
     - ``changepoint_lab.SDHMM``
     - ``0.3.0``
   * - ``sdhmm_mix_vi``
     - ``changepoint_lab.SDHMMMixVI``
     - ``0.3.0``
   * - ``within_period``
     - ``changepoint_lab.WithinPeriodCPD``
     - ``0.3.0``

Migration Notes
---------------

Use ``changepoint_lab.__stable__`` to audit stable package-level imports and
``changepoint_lab.__experimental__`` to identify opt-in implementations. The
older lowercase aliases above remain available only through the compatibility
layer and should not be used in new code.

The stale ``changepoint_lab.bocpd`` alias is not provided. Use
``changepoint_lab.BOCPD`` or ``changepoint_lab.algorithms.bayesian.bocpd.BOCPD``.
