SD-HMM Module
=============

The :mod:`sdhmm` package implements Hidden Markov Models with
Scaled-Dirichlet emissions for compositional data [1]_.

Core Classes
------------

.. autoclass:: sdhmm.sdhmm.SDHMMConfig
   :members:

.. autoclass:: sdhmm.sdhmm.SDHMMResult
   :members:

.. autoclass:: sdhmm.sdhmm.SDHMM
   :members:

Mixture Extensions
------------------

.. automodule:: sdhmm.sdhmm_mix_vi
   :members:

Utilities
---------

.. automodule:: sdhmm.sdhmm
   :members: _as_float_array, _normalize_rows_stable

Example
-------

.. code-block:: python

    import numpy as np
    from sdhmm.sdhmm import SDHMM, SDHMMConfig

    X = np.random.dirichlet(np.ones(3), size=200)
    model = SDHMM(SDHMMConfig(K=2))
    result = model.fit(X)
    print(result.loglik)

References
----------

.. [1] Manouchehri, M. & Bouguila, N. (2023).
       *Hidden Markov models with scaled Dirichlet distributions for compositional data*.
