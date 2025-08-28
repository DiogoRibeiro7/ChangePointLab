import numpy as np
from bayesian_blocks import bayesian_blocks_bernoulli
from bb_plotting import plot_blocks_index
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
N = 300
p = np.r_[np.full(120, 0.2), np.full(180, 0.6)]
x = rng.binomial(1, p)

# Use successes = x, trials = 1 each
res = bayesian_blocks_bernoulli(successes=x, trials=None, p0=0.05)
ax = plot_blocks_index(N=N, result=res, ylabel="p", title="Bernoulli blocks")
plt.show()
