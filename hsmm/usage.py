import numpy as np
from gaussian_diag import estimate_by_kmeanspp, gaussian_diag_loglik
from hsmm import HSMM, HSMMConfig, HSMMParams, PoissonDur

rng = np.random.default_rng(0)
T, D, K = 1200, 5, 3

# Example multivariate data (three regimes)
means = np.array([[0,0,0,0,0],[2,-1,1,0.5,0],[ -1.5,1.0,-0.5,0.5,1.0 ]])
seg = np.repeat([0,1,2],[400,400,400])
X = rng.normal(0,1,(T,D)) + means[seg]

# 1) Initialize emissions from X (no labels needed)
em_params = estimate_by_kmeanspp(X, K, n_init=5, max_iter=100, allow_nan=False)

# 2) Build log-likelihoods for HSMM
L = gaussian_diag_loglik(X, em_params)

# 3) Fit HSMM durations + transitions (emissions fixed)
pi0 = np.full(K, 1.0/K)
A0 = np.full((K,K), 1.0/(K-1)); np.fill_diagonal(A0, 0.0)
dur = ("poisson", PoissonDur(lam=np.array([60.0, 80.0, 70.0])))

model = HSMM(HSMMConfig(K=K, Dmax=150, min_duration=5), HSMMParams(pi=pi0, A=A0, duration=dur))
params_fit, ll_trace = model.fit(L)
