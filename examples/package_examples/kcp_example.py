import numpy as np
import matplotlib.pyplot as plt
from changepoint_lab.algorithms.kernel.kcp_core import (
    gram_rbf,
    gram_linear,
    build_kernel_prefix,
    kcp_penalized,
    kcp_select_bic,
)
from changepoint_lab.common.plotting.kcp_plotting import (
    plot_segments_1d,
    plot_model_scree,
)

rng = np.random.default_rng(0)

# Build a 2D sequence with three regimes (mean shifts)
n1, n2, n3 = 200, 160, 220
X = np.vstack([
    rng.normal([0.0, 0.0], 0.8, size=(n1, 2)),
    rng.normal([2.0, -1.0], 0.8, size=(n2, 2)),
    rng.normal([-1.5, 1.5], 0.8, size=(n3, 2)),
])

# Use RBF kernel (median heuristic) and build prefix structures
K, _ = gram_rbf(X)       # or gram_linear(X)
pref = build_kernel_prefix(K)

# Penalized fit with PELT (expected linear time)
res = kcp_penalized(pref, penalty=np.log(X.shape[0]), min_size=20, method="pelt")
print("Change points (penalized):", res.change_points.tolist())

# Fixed-m + BIC-style model selection
sel = kcp_select_bic(pref, m_max=8, beta=1.0, min_size=20)
print("Selected m*:", sel.m_star, "CPs:", sel.change_points.tolist())

# Plots (use a 1D projection for display)
x_proj = X[:, 0]  # or your favorite projection
plot_segments_1d(x_proj, sel.edges, title="KCP (RBF) with BIC-style selection")
plt.show()

plot_model_scree(sel)
plt.show()
