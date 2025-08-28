import numpy as np
from bayesian_blocks import bayesian_blocks_counts
from bb_plotting import plot_blocks_index
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
N = 200
rate = np.r_[np.full(80, 3.0), np.full(120, 0.8)]
counts = rng.poisson(rate)
res = bayesian_blocks_counts(counts, widths=None, p0=0.05)
ax = plot_blocks_index(N=N, result=res, ylabel="rate", title="Binned Poisson")
plt.show()
