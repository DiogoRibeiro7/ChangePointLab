BOCPD Module
============

The :mod:`bocpd` package implements Bayesian Online Changepoint Detection
for binary data [1]_.  It provides configurable hazard functions, flexible
prior settings and plotting helpers.

Configuration and Result Types
------------------------------

.. autoclass:: bocpd.bocpd.BOCPDConfig
   :members:
   :show-inheritance:

.. autoclass:: bocpd.bocpd.BOCPDResult
   :members:
   :show-inheritance:

Detector
--------

.. autoclass:: bocpd.bocpd.BOCPD
   :members:
   :show-inheritance:

Hazard Functions
----------------

.. autoclass:: bocpd.bocpd.ConstantHazard
   :members:

.. autoclass:: bocpd.bocpd.ScheduledHazard
   :members:

.. autoclass:: bocpd.bocpd.BoostedBoundaryHazard
   :members:

Validation Helpers
------------------

.. automodule:: bocpd.validation
   :members:

Plotting Utilities
------------------

.. automodule:: bocpd.bocpd_plotting
   :members:

Example
-------

.. code-block:: python

    from bocpd.bocpd import BOCPD, BOCPDConfig, ConstantHazard
    x = [0, 0, 1, 1, 1, 0]
    cfg = BOCPDConfig(max_run_length=200)
    model = BOCPD(ConstantHazard(100), cfg)
    result = model.run(x)
    print(result.cp_prob)

Related Components
------------------

Offline segmentation is available via :mod:`pelt.pelt` and energy
based detection via :mod:`edivisive.edivisive`.

References
----------

.. [1] Adams, R. P. & MacKay, D. J. C. (2007).
       *Bayesian Online Changepoint Detection*.
