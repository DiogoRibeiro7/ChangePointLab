import numpy as np

from edivisive.edivisive import edivisive


def test_edivisive_memmap_and_deque():
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    res = edivisive(x, min_size=5, R=9, seed=0, chunk_size=10, use_memmap=True)
    assert res.n == 60
    assert res.labels.shape == (60,)
