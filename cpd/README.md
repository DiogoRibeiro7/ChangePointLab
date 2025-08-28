# Change-Point & State-Space Toolkit (NumPy‑only)

Lightweight algorithms for offline/online change‑point detection and hidden‑state modeling. Everything is written with a focus on clarity, numerical stability, and minimal dependencies (NumPy + Matplotlib for plots). Good defaults, type hints, docstrings, and small plotting/utility helpers are included.

## ✨ What’s inside

* **Bayesian Blocks** (`bayesian_blocks.py`, `bb_plotting.py`, `bb_utils.py`)

  * Exact DP for **events** (unbinned Poisson), **binned Poisson counts**, and **Bernoulli/Binomial** streams.
  * Scargle FPR prior (`p0`) → penalty `γ`; fast O(N²) solver with clean plots.
* **E‑Divisive** (`edivisive.py`, `edivisive_plotting.py`)

  * Multivariate energy‑statistic segmentation with **permutation test**.
  * **Block‑bootstrap** options for short‑range dependence: `iid`, `block-permutation`, `circular-block-bootstrap`.
* **Kernel Change‑Point Detection (KCP)** (`kcp.py`, `kcp_plotting.py`)

  * Linear/RBF kernels, DP for fixed‑m and penalized fits, **PELT** pruning.
  * BIC‑style model selection over `m`.
* **RFF KCP** (`kcp_rff.py`)

  * **Random Fourier Features** embedding for RBF to avoid the full Gram; least‑squares CPD + PELT.
* **Scaled‑Dirichlet HMMs**

  * `sdhmm.py` — single SD emission per state (MAP/variational hybrid).
  * `sdhmm_mix_vi.py` — **per‑state mixtures** with VI for mixture weights & emissions.
* **Explicit‑Duration HSMM** (`hsmm.py`)

  * Forward–backward and Viterbi with **Poisson** or **NegBin** dwell times (explicit durations), EM updates.
* **Gaussian Diagonal Emissions** (`gaussian_diag.py`)

  * Build `loglik_tk` for HSMM/HMM, estimators from hard/soft labels, and a tiny **k‑means++** initializer (NumPy‑only).

Tested on Python **3.9+**.

---

## Installation

```bash
# clone your repo
pip install -U numpy matplotlib
```

That’s it. The code uses only NumPy (and Matplotlib for optional plots).

---

## Quickstart

### 1) Kernel CPD (RBF) with PELT

```python
import numpy as np
from kcp import gram_rbf, build_kernel_prefix, kcp_penalized

X = ...  # (n, d) array
K, gamma = gram_rbf(X)                   # median heuristic if gamma not passed
pref = build_kernel_prefix(K)
res = kcp_penalized(pref, gamma=np.log(X.shape[0]), min_size=20, method="pelt")
print(res.change_points)
```

### 1b) RFF KCP (no full Gram)

```python
from kcp_rff import RFFConfig, rbf_rff_map, build_feature_prefix, rff_kcp_penalized

rff = rbf_rff_map(X, RFFConfig(n_features=512, seed=123))
pref = build_feature_prefix(rff.Z)
res = rff_kcp_penalized(pref, gamma_pen=np.log(X.shape[0]), min_size=20, method="pelt")
```

### 2) Bayesian Blocks

**Events (unbinned Poisson):**

```python
from bayesian_blocks import bayesian_blocks_events
res = bayesian_blocks_events(t, t_start=0.0, t_stop=10.0, p0=0.05)
print(res.edges, res.block_value)
```

**Binned counts:**

```python
from bayesian_blocks import bayesian_blocks_counts
res = bayesian_blocks_counts(counts, widths=None, p0=0.05)
```

**Bernoulli / binary stream:**

```python
from bayesian_blocks import bayesian_blocks_bernoulli
res = bayesian_blocks_bernoulli(successes=x, trials=None, p0=0.05)
```

### 3) E‑Divisive (with block bootstrap)

```python
from edivisive import edivisive
res = edivisive(X, alpha=1.0, min_size=30, R=499,
                resample="circular-block-bootstrap", block_size=None,
                significance=0.05, seed=123, progress=True)
print(res.change_points)
```

### 4) HSMM with explicit durations (Poisson) + GaussianDiag emissions

```python
import numpy as np
from hsmm import HSMM, HSMMConfig, HSMMParams, PoissonDur
from gaussian_diag import estimate_by_kmeanspp, gaussian_diag_loglik

X = ...  # (T, D) observations
K = 3
em = estimate_by_kmeanspp(X, K, n_init=5, allow_nan=False)
L = gaussian_diag_loglik(X, em)

pi0 = np.full(K, 1.0/K)
A0 = np.full((K,K), 1.0/(K-1)); np.fill_diagonal(A0, 0.0)  # HSMM transitions
hsmm = HSMM(HSMMConfig(K=K, Dmax=150, min_duration=5),
            HSMMParams(pi=pi0, A=A0, duration=("poisson", PoissonDur(lam=np.array([60,80,70])))))
params_fit, ll_trace = hsmm.fit(L)
z_vit, d_vit = hsmm.decode_viterbi(L)
```

---

## API at a glance

### Bayesian Blocks

* `bayesian_blocks_events(t, t_start=None, t_stop=None, p0=0.05, gamma=None)`
* `bayesian_blocks_counts(counts, widths=None, p0=0.05, gamma=None)`
* `bayesian_blocks_bernoulli(successes, trials=None, p0=0.05, gamma=None)`
* Helpers: `blocks_to_labels_index`, `blocks_to_labels_time`; plotting in `bb_plotting.py`.

### E‑Divisive

* `edivisive(X, alpha=1.0, min_size=20, R=499, significance=0.05, resample='iid'|'block-permutation'|'circular-block-bootstrap', block_size=None, seed=123)`

### Kernel CPD (exact Gram)

* `gram_linear(X)` → `K`
* `gram_rbf(X, gamma=None, sigma=None)` → `(K, gamma)`
* `build_kernel_prefix(K)` → prefix sums for O(1) segment costs
* Penalized: `kcp_penalized(pref, gamma, min_size=1, method='pelt'|'op', grid_jump=1)`
* Fixed‑m: `kcp_fixed_m(pref, m, min_size=1, grid_jump=1)`
* Model selection: `kcp_select_bic(pref, m_max, beta=1.0, min_size=1, grid_jump=1)`

### RFF CPD (no Gram)

* `rbf_rff_map(X, RFFConfig(...))` → `RFFMap(Z, gamma, W, b)`
* `build_feature_prefix(Z)` → O(1) segment costs in feature space
* Penalized: `rff_kcp_penalized(pref, gamma_pen, min_size=1, method='pelt'|'op', grid_jump=1)`
* Fixed‑m: `rff_kcp_fixed_m(pref, m, min_size=1, grid_jump=1)`

### SD‑HMM / Mixture SD‑HMM

* `SDHMM(SDHMMConfig).fit(X)` and `.viterbi(X)`
* `SDHMMMixVI(SDHMMMixVIConfig).fit(X)` and decoders for states/components

### HSMM

* `HSMM(HSMMConfig, HSMMParams).fit(loglik_tk)` → updates durations & transitions
* `decode_viterbi(loglik_tk)` → most likely state path with explicit durations

### GaussianDiag emissions

* `gaussian_diag_loglik(X, GaussianDiagParams, allow_nan=False)` → `(T, K)` log‑liks
* Estimators: `estimate_from_labels`, `estimate_from_responsibilities`, `estimate_by_kmeanspp`

---

## Design & numerics

* **Log‑space** forward–backward (HSMM) and scaled operations to avoid underflow.
* Stable digamma/softmax updates in SD‑HMM emissions; mirror‑descent step for simplex constraints.
* **Prefix sums** (1D and 2D) for O(1) segment cost queries in BB / KCP / RFF.
* **PELT** pruning for expected linear time on penalized fits.
* Careful clamping (`eps`) for logs, divisions, and variance floors.

---

## Complexity (rough)

* Bayesian Blocks: O(N²) DP (use pre‑binning/decimation for very large N).
* E‑Divisive: O(R · m²) per tested segment (distance matrix + prefix sums), divisive recursion.
* KCP (Gram): O(n²) to build K, OP is O(n²), PELT prunes in practice.
* KCP (RFF): O(n·d·D) to embed, then OP/PELT over O(n·D) prefix features.
* HSMM: O(T · K · D\_max + T · K²).

---

## Practical tips

* **Penalties**: start with `γ ≈ c·log n` (c∈\[0.5,2]) for penalized DP; tune by scree or held‑out error.
* **RBF bandwidth**: median heuristic works well; for RFF you can pass `sigma`/`gamma` if you know the scale.
* **E‑Divisive resampling**: use `circular-block-bootstrap` with `block_size` ≈ 10–30 for weak AR(1)‑like dependence; increase if persistence is stronger.
* **HSMM durations**: set realistic `min_duration` and `Dmax` near the 95–99th percentile of dwell times.
* **GaussianDiag**: `estimate_by_kmeanspp` is a fast emission initializer when labels are unknown.

---

## Project structure

```
bayesian_blocks.py
bb_plotting.py
bb_utils.py
edivisive.py
edivisive_plotting.py
kcp.py
kcp_plotting.py
kcp_rff.py
sdhmm.py
sdhmm_mix_vi.py
hsmm.py
gaussian_diag.py
```

---

## Roadmap

* CLI wrappers for each method (CSV in → results + PNG/SVG plots + CSV of edges/labels).
* Additional emissions (e.g., full‑covariance Gaussians, AR‑emissions for HSMM).
* RFF variants (orthogonal/Quasi‑MC) and automatic bandwidth cross‑validation.

---

## License

MIT License. See headers in source files.

---

## References (primary)

* Scargle, J. D. (2013): *Studies in Astronomical Time Series Analysis. VI. Bayesian Block Representations.*
* Matteson, D. S., & James, N. A. (2014): *A Nonparametric Approach for Multiple Change Point Analysis of Multivariate Data.*
* Harchaoui, Z., & Cappé, O. (2007–2010): *Retrospective multiple change-point estimation with kernels.*
* Arlot, S., Celisse, A., & Harchaoui, Z. (2019): *A kernel multiple change-point algorithm via model selection.*
* Rahimi, A., & Recht, B. (2008): *Random Features for Large-Scale Kernel Machines.*
* Yu, S. Z. (2010): *Hidden semi-Markov models.*
* Johnson, M. J., & Willsky, A. S. (2013): *Bayesian Nonparametric HSMMs.*
