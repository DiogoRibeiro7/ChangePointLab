Within-Period Detection Module
==============================

The :mod:`within_period` package implements changepoint detection on
periodic domains using reversible-jump MCMC.

Core Classes
------------

.. autoclass:: within_period.within_period_cpd.WithinPeriodCPD
   :members:

.. autoclass:: within_period.within_period_cpd.RJConfig
   :members:

.. autoclass:: within_period.within_period_cpd.ModelPrior
   :members:

RJMCMC Samplers
---------------

.. automodule:: within_period.samplers.tempering
   :members:

Posterior Sampling and Summaries
--------------------------------

.. automodule:: within_period.posterior_predictive
   :members:

Visualization Helpers
---------------------

.. automodule:: within_period.anchor_utils
   :members:

Example
-------

.. code-block:: python

    import numpy as np
    from within_period.within_period_cpd import WithinPeriodCPD, ModelPrior

    prior = ModelPrior(N=24, alpha=1.0, beta=1.0, min_seg_len=2)
    model = WithinPeriodCPD(prior)
    x = np.random.binomial(1, 0.2, size=240)
    result = model.fit(x)
    print(result.changepoint_hist)

References
----------

.. [1] Green, P. J. (1995). *Reversible jump Markov chain Monte Carlo
       computation and Bayesian model determination*.
