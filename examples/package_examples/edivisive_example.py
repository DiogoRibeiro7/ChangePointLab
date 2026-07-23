import numpy as np
import matplotlib.pyplot as plt
from changepoint_lab import edivisive
from changepoint_lab.common.plotting.edivisive_plotting import (
    plot_scree_edivisive,
    plot_segments_1d,
)

rng = np.random.default_rng(0)

# Build a 1D sequence with two distribution changes (location + scale)
n1, n2, n3 = 300, 250, 350
x = np.r_[rng.normal(0.0, 1.0, n1),
          rng.normal(2.0, 0.7, n2),
          rng.normal(-1.5, 1.2, n3)]
X = x  # 1D; for multivariate use shape (n,d)

res = edivisive(X, alpha=1.0, min_size=40, R=499, significance=0.05, seed=123, progress=True)

print("Change points:", res.change_points.tolist())
print("Segments:", np.unique(res.labels).size)

# Plots
plot_scree_edivisive(res)
plt.show()

plot_segments_1d(x, res, title="E-Divisive (1D)")
plt.show()




rng = np.random.default_rng(0)
n = 1500
# weak AR(1)-like dependence + two distribution shifts
x = np.zeros(n)
eps = rng.normal(0, 1, size=n)
phi = 0.3
for t in range(1, n):
    x[t] = phi * x[t-1] + eps[t]
x[500:] += 1.5
x[1000:] -= 2.0

# 1) CBB with automatic block size
res_cbb = edivisive(x, alpha=1.0, min_size=30, R=399,
                    resample="circular-block-bootstrap", block_size=None,
                    seed=123, progress=True)

# 2) Non-overlapping block permutation with b=25
res_bp = edivisive(x, alpha=1.0, min_size=30, R=399,
                   resample="block-permutation", block_size=25,
                   seed=123, progress=True)
