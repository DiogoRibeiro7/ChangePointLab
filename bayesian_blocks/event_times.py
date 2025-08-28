import numpy as np
from bayesian_blocks import bayesian_blocks_events
from bb_plotting import plot_blocks_time
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
# piecewise-constant rate: 2.0 until t=5, then 0.5 until t=10
t1 = np.cumsum(rng.exponential(1/2.0, size=120))
t1 = t1[t1 <= 5.0]
t2 = 5.0 + np.cumsum(rng.exponential(1/0.5, size=120))
t2 = t2[t2 <= 10.0]
t = np.sort(np.concatenate([t1, t2]))

res = bayesian_blocks_events(t, t_start=0.0, t_stop=10.0, p0=0.05)
ax = plot_blocks_time(t_min=0.0, t_max=10.0, result=res, title="Events: Poisson rate")
plt.show()
