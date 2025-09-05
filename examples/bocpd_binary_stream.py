import numpy as np
import matplotlib.pyplot as plt
from changepoint_lab.algorithms.bayesian.bocpd import BOCPD, BOCPDConfig, ConstantHazard

rng = np.random.default_rng(0)
N = 300
p = np.r_[np.full(120, 0.2), np.full(180, 0.6)]
x = rng.binomial(1, p)

hazard = ConstantHazard(mean_run_length=100)
model = BOCPD(hazard, cfg=BOCPDConfig())
result = model.fit(x).predict()

plt.step(np.arange(N), x, where="post", label="data")
for cp in result.indices:
    plt.axvline(cp, color="r", linestyle="--")
plt.ylabel("x")
plt.title("BOCPD on Bernoulli sequence")
plt.show()
