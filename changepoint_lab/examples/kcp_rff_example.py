import numpy as np
import matplotlib.pyplot as plt
from changepoint_lab.algorithms.kernel.kcp_rff import (
    RFFConfig,
    rbf_rff_map,
    build_feature_prefix,
    rff_kcp_penalized,
    rff_kcp_fixed_m,
)

rng = np.random.default_rng(0)

# Synthetic 3-regime 4D data
n1, n2, n3 = 220, 180, 240
X = np.vstack([
    rng.normal(0.0, 1.0, size=(n1, 4)),
    rng.normal([2.0, -1.0, 0.5, 0.0], 1.0, size=(n2, 4)),
    rng.normal([-1.5, 1.0, -0.5, 1.0], 1.0, size=(n3, 4)),
])

# 1) Map to RFF (no full Gram); D=512 features; gamma via subsampled median heuristic
rff = rbf_rff_map(X, RFFConfig(n_features=512, gamma=None, subsample_for_bandwidth=1500, seed=123))

# 2) Prefix sums and penalized PELT
pref = build_feature_prefix(rff.Z)
res = rff_kcp_penalized(pref, gamma_pen=np.log(X.shape[0]), min_size=25, method="pelt")

print("RFF gamma used:", rff.gamma)
print("Change points (PELT):", res.change_points.tolist())

# 3) Fixed-m (e.g., m=3 segments) if you want the exact m
sn = rff_kcp_fixed_m(pref, m=3, min_size=25)
print("Fixed-m edges:", sn.edges.tolist())

# Quick 1D visualization (project first coordinate)
plt.plot(np.arange(X.shape[0]), X[:, 0], lw=1.0)
for e in res.edges[1:-1]:
    plt.axvline(int(e), ls="--")
plt.title("RFF KCP (RBF) — PELT segmentation")
plt.show()
