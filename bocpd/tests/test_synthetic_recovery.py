import numpy as np
from bocpd import BOCPD, BOCPDConfig, ConstantHazard

def _synth_piecewise(n, cps, p_lo=0.05, p_hi=0.25, seed=1):
    rng = np.random.default_rng(seed)
    x = np.empty(n, dtype=bool)
    last = 0
    p = p_lo
    for c in cps + [n]:
        x[last:c] = rng.binomial(1, p, size=c - last).astype(bool)
        p = p_hi if p == p_lo else p_lo
        last = c
    return x

def test_recovery_precision_recall_window():
    n = 3000
    true_cps = [800, 1600, 2400]
    x = _synth_piecewise(n, true_cps)

    cfg = BOCPDConfig(alpha0=0.5, beta0=0.5, max_run_length=512, store_run_length_posterior=False)
    hazard = ConstantHazard(mean_run_length=400.0)
    m = BOCPD(cfg=cfg, hazard=hazard)
    res = m.run(x)

    thr = 0.6
    detected = np.flatnonzero(res.cp_prob >= thr)
    # tolerance window
    win = 10

    tp = 0
    matched = set()
    for d in detected:
        if any(abs(d - t) <= win for t in true_cps):
            tp += 1
            matched.add(min(true_cps, key=lambda t: abs(d - t)))
    precision = tp / max(1, len(detected))
    recall = len(matched) / len(true_cps)

    assert precision >= 0.5
    assert recall >= 0.5
