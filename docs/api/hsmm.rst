HSMM Module
===========

The :mod:`hsmm` package provides explicit-duration hidden Markov models
with flexible emission models and duration distributions [1]_.

Configuration and Parameters
----------------------------

.. autoclass:: hsmm.hsmm.HSMMConfig
   :members:

.. autoclass:: hsmm.hsmm.HSMMParams
   :members:

Duration Models
---------------

.. autoclass:: hsmm.hsmm.PoissonDur
   :members:

.. autoclass:: hsmm.hsmm.NegBinDur
   :members:

Emission Models
---------------

.. automodule:: hsmm.gaussian_diag
   :members:

.. automodule:: hsmm.gaussian_full
   :members:

.. automodule:: hsmm.ar_emissions
   :members:

Core Algorithms
---------------

.. autoclass:: hsmm.hsmm.HSMM
   :members:
   :show-inheritance:

Example
-------

.. code-block:: python

    from hsmm.hsmm import HSMM, HSMMConfig, HSMMParams, PoissonDur
    import numpy as np

    cfg = HSMMConfig(K=2, Dmax=50)
    params = HSMMParams(pi=np.array([0.5,0.5]),
                        A=np.array([[0.0,1.0],[1.0,0.0]]),
                        duration=("poisson", PoissonDur(np.array([10.,10.]))))
    model = HSMM(cfg, params)
    loglik = np.random.randn(100,2)
    result = model.decode_viterbi(loglik)

References
----------

.. [1] Yu, S.-Z. (2010). *Hidden semi-Markov models.*
.. [2] Johnson, M. & Willsky, A. (2013). *Bayesian nonparametric hidden
       semi-Markov models.*
