import numpy as np
from changepoint_lab import SDHMMMixVI, SDHMMMixVIConfig

# Synthetic proportional data: T x D
rng = np.random.default_rng(0)
T, D, K, M = 1500, 8, 3, 2
X_raw = rng.random((T, D))

model = SDHMMMixVI(SDHMMMixVIConfig(K=K, M=M, max_iter=100, min_iter=5, tol=1e-5))
model.fit(X_raw)
res = model.result_

print("ELBO proxy (sequence loglik):", res.loglik)
print("pi:", res.pi.round(3))
print("A row 0:", res.A[0].round(3))
print("eta (state 0):", res.eta[0].round(3))
print("alpha(state0,comp0)[:4]:", res.params[0][0].alpha[:4].round(3))
print("beta(state0,comp0)[:4]:", res.params[0][0].beta[:4].round(3))

z_hat = model.viterbi_states(X_raw)
m_hat = model.most_likely_components(X_raw)
