import numpy as np
from sdhmm import SDHMM, SDHMMConfig

# X must be non-negative; rows will be renormalized to sum to 1 (proportions).
# Example: 3 states on D=6 features
rng = np.random.default_rng(0)
T, D, K = 2000, 6, 3
X_raw = rng.random((T, D))
model = SDHMM(SDHMMConfig(K=K, max_iter=100, min_iter=5, tol=1e-5))
res = model.fit(X_raw)

print("log-likelihood:", res.loglik)
print("pi:", res.pi.round(3))
print("A:", res.A.round(3))
print("alpha[0]:", res.params[0].alpha.round(3))
print("beta[0]:", res.params[0].beta.round(3))

# Decode most likely states
z_hat = model.viterbi(X_raw)
