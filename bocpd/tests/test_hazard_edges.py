import numpy as np
from bocpd import ConstantHazard, ScheduledHazard, BoostedBoundaryHazard

def test_scheduled_hazard_clipped():
    sched = np.array([1e-9, 0.2, 0.8, 0.999999], dtype=float)
    h = ScheduledHazard(schedule=sched, period=4)
    vals = [h.prob(0, t) for t in range(8)]
    assert all(0.0 < v < 1.0 for v in vals)

def test_boost_never_reaches_one():
    base = ConstantHazard(mean_run_length=1000.0)
    h = BoostedBoundaryHazard(base=base, boundaries={0}, period=96, boost_factor=500.0)
    boosted = h.prob(0, 96)  # boundary
    assert 0.0 < boosted < 1.0
