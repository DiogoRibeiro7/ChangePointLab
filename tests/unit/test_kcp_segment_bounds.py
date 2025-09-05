import numpy as np
import kcp


def test_kcp_respects_max_seg_len():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 1))
    K = kcp.gram_linear(X)
    pref = kcp.build_kernel_prefix(K)
    res = kcp.kcp_penalized(pref, penalty=0.0, min_size=1, method="op", max_seg_len=2)
    lengths = np.diff(res.edges)
    assert np.all(lengths <= 2)


def test_kcp_handles_large_grid_jump():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 1))
    K = kcp.gram_linear(X)
    pref = kcp.build_kernel_prefix(K)
    res = kcp.kcp_penalized(pref, penalty=1.0, min_size=3, grid_jump=50)
    assert res.change_points.size == 0
