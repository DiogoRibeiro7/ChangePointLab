E-Divisive Module
=================

The :mod:`edivisive` package implements the hierarchical E-Divisive
with Medians procedure for non-parametric changepoint detection [1]_.

Main Algorithm
--------------

.. autofunction:: edivisive.edivisive.edivisive

Distance and Permutation Utilities
----------------------------------

.. autofunction:: edivisive.edivisive._pairwise_energy_dist_alpha

.. autofunction:: edivisive.edivisive._resample_iid_permutation

.. autofunction:: edivisive.edivisive._resample_block_permutation

.. autofunction:: edivisive.edivisive._resample_circular_block_bootstrap

Result Containers
-----------------

.. autoclass:: edivisive.edivisive.EDivisiveSplit
   :members:

.. autoclass:: edivisive.edivisive.EDivisiveResult
   :members:

Visualization
-------------

.. automodule:: edivisive.edivisive_plotting
   :members:

Example
-------

.. code-block:: python

    import numpy as np
    from edivisive.edivisive import edivisive

    X = np.concatenate([np.random.randn(50), np.random.randn(50)+2])[:, None]
    res = edivisive(X, alpha=1.0, min_size=10, n_perm=100)
    print(res.change_points)

References
----------

.. [1] Matteson, D. S. & James, N. A. (2014).
       *A Nonparametric Approach for Multiple Change Point Analysis of Multivariate Data*.
