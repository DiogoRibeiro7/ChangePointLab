import numpy as np

from algorithms.optimization.cost_functions import NormalMeanVarUnknown
from algorithms.optimization.pelt import pelt
from edivisive.edivisive import edivisive
from within_period.within_period_cpd import ModelPrior, WithinPeriodCPD, RJConfig


def test_methods_interoperability():
    """Run multiple detectors on the same series to ensure basic interoperability."""
    data = np.concatenate([np.zeros(40), np.ones(40)])

    # PELT with Gaussian cost
    cost = NormalMeanVarUnknown()
    pelt_res = pelt(data, cost_fn=cost, penalty=1.0, min_seg_len=5)

    # E-Divisive nonparametric
    ediv_res = edivisive(data, min_size=5, R=10, significance=0.1)

    # Within-period CPD on binary data
    prior = ModelPrior(N=20, l=5)
    wp = WithinPeriodCPD(prior)
    cfg = RJConfig(iters=20, burn=5, thin=5, seed=0)
    wp_res = wp.fit((data > 0).astype(int), cfg=cfg)

    assert 40 in pelt_res.change_points
    assert 40 in ediv_res.change_points
    assert wp_res.samples_tau is not None
