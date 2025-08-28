import numpy as np
from bocpd import BOCPD, BOCPDConfig, ConstantHazard

def test_online_equals_batch():
    rng = np.random.default_rng(0)
    x = rng.binomial(1, 0.1, size=1000).astype(bool)

    cfg = BOCPDConfig(alpha0=1.0, beta0=1.0, max_run_length=256, store_run_length_posterior=False)
    hazard = ConstantHazard(mean_run_length=96.0)
    m1 = BOCPD(cfg=cfg, hazard=hazard)
    res = m1.run(x)

    m2 = BOCPD(cfg=cfg, hazard=hazard)
    cp = []
    mr = []
    pm = []
    for xi in x:
        out = m2.update(bool(xi))
        cp.append(out["cp_prob"])
        mr.append(out["map_run_length"])
        pm.append(out["pred_mean"])

    assert np.allclose(np.array(cp), res.cp_prob, atol=1e-12)
    assert np.array_equal(np.array(mr), res.map_run_length)
    assert np.allclose(np.array(pm), res.pred_mean, atol=1e-12)
