PELT Module
===========

The :mod:`pelt` module implements the Pruned Exact Linear Time algorithm
for offline changepoint detection [1]_.

Main Algorithm
--------------

.. autofunction:: pelt.pelt.pelt

Cost Functions
--------------

.. autoclass:: pelt.pelt.NormalMeanKnownVar
   :members:

.. autoclass:: pelt.pelt.NormalMeanVarUnknown
   :members:

.. autoclass:: pelt.pelt.BetaBinomialCost
   :members:

Penalty Helpers
---------------

.. autofunction:: pelt.pelt.bic_penalty

.. autofunction:: pelt.pelt.aic_penalty

Result Container
----------------

.. autoclass:: pelt.pelt.PELTResult
   :members:

Example
-------

.. code-block:: python

    import numpy as np
    from pelt.pelt import pelt, NormalMeanVarUnknown, bic_penalty

    data = np.r_[np.zeros(50), np.ones(50)]
    cost = NormalMeanVarUnknown()
    cost.precompute(data)
    res = pelt(data, cost, penalty=bic_penalty(2, len(data)))
    print(res.change_points)

Related Components
------------------

The online counterpart is :class:`bocpd.bocpd.BOCPD`.

References
----------

.. [1] Killick, R., Fearnhead, P. & Eckley, I. A. (2012).
       *Optimal detection of changepoints with a linear computational cost*.
