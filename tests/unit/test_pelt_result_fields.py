import numpy as np

from pelt.pelt import NormalMeanVarUnknown, pelt


def test_pelt_result_contains_labels_and_costs():
    data = np.concatenate([np.zeros(50), np.ones(50)])
    cost_fn = NormalMeanVarUnknown()
    res = pelt(data, cost_fn=cost_fn, penalty=1.0, min_seg_len=10)
    assert res.labels.shape == (len(data),)
    assert res.costs_per_segment.shape == (len(res.change_points) + 1,)
    # labels should enumerate contiguous segments
    assert set(res.labels) == {0, 1}
