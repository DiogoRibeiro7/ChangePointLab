# Scientific Method Registry

Date: 2026-07-23

This table renders `docs/science/method_registry.yml`. It distinguishes
faithful reproductions, adaptations, extensions, heuristics, and missing methods
before scientific behavior is changed.

| ID | Method | Source status | Primary citations | Stable package symbols | Code paths | Verification |
| --- | --- | --- | --- | --- | --- | --- |
| `pelt` | Pruned Exact Linear Time | adaptation | `killick_2012` | `PELT` | `src/changepoint_lab/algorithms/optimization/pelt.py`; `src/changepoint_lab/algorithms/optimization/cost_functions.py` | partially_verified |
| `bocpd_beta_bernoulli` | Bayesian Online Changepoint Detection, Beta-Bernoulli and Poisson-Gamma paths | adaptation | `adams_mackay_2007` | `BOCPD`, `BOCPDConfig`, `BOCPDAlertConfig`, `BOCPDResult`, `BetaBernoulli`, `PoissonGamma`, hazards | `src/changepoint_lab/algorithms/bayesian/bocpd/core.py`; `src/changepoint_lab/algorithms/bayesian/bocpd/likelihoods.py`; `src/changepoint_lab/algorithms/bayesian/bocpd/__init__.py` | partially_verified |
| `within_period_rjmcmc` | Within-period changepoint detection | adaptation | `taylor_2021` | `WithinPeriodCPD` | `src/changepoint_lab/algorithms/bayesian/within_period/`; `scripts/run_within_period_reproduction.py` | partially_verified |
| `sliced_poisson_process` | Sliced Poisson process changepoint detection | adaptation | `martinez_hernandez_killick_2024` | `SlicedPoissonCPD`, `SlicedPoissonConfig`, `EventPeriod` | `src/changepoint_lab/algorithms/point_process/sliced_poisson.py` | partially_verified |
| `edivisive` | E-Divisive energy-statistic changepoint detection | adaptation | `matteson_james_2014` | `EDivisive`, `edivisive` | `src/changepoint_lab/algorithms/nonparametric/edivisive.py`; `src/changepoint_lab/algorithms/nonparametric/edivisive_core.py` | partially_verified |
| `kernel_cpd` | Kernel changepoint detection | adaptation | `harchaoui_bach_2008`, `arlot_celisse_harchaoui_2019` | `KernelCPD`, `gram_rbf`, `kcp_penalized`, `kcp_select_bic` | `src/changepoint_lab/algorithms/kernel/kcp.py`; `src/changepoint_lab/algorithms/kernel/kcp_core.py`; `src/changepoint_lab/algorithms/kernel/bandwidth_cv.py` | characterized_only |
| `rff_kernel_cpd` | Random Fourier feature kernel changepoint detection | extension | `rahimi_recht_2007`, `harchaoui_bach_2008` | None | `src/changepoint_lab/algorithms/kernel/kcp_rff.py`; `src/changepoint_lab/algorithms/kernel/rff_variants.py` | partially_verified |
| `hsmm` | Hidden semi-Markov model | adaptation | `yu_2010` | `HSMM`, `HSMMConfig`, `HSMMParams`, `PoissonDur` | `src/changepoint_lab/algorithms/state_space/hsmm.py`; `src/changepoint_lab/algorithms/state_space/hsmm_core.py` | partially_verified |
| `sd_hmm` | Scaled-Dirichlet HMM | adaptation | `manouchehri_bouguila_2023` | `SDHMM`, `SDHMMConfig`, `SDHMMResult` | `src/changepoint_lab/algorithms/state_space/sdhmm.py` | characterized_only |
| `sd_hmm_mix_vi` | Scaled-Dirichlet mixture VI HMM | extension | `manouchehri_bouguila_2023` | `SDHMMMixVI`, `SDHMMMixVIConfig`, `SDHMMMixVIResult` | `src/changepoint_lab/algorithms/state_space/sdhmm_mix_vi.py` | unverified |

## Citation Identifiers

| ID | Citation | DOI or archival URL |
| --- | --- | --- |
| `adams_mackay_2007` | Adams and MacKay (2007), Bayesian Online Changepoint Detection | https://arxiv.org/abs/0710.3742 |
| `killick_2012` | Killick, Fearnhead, and Eckley (2012), Optimal Detection of Changepoints With a Linear Computational Cost | https://doi.org/10.1080/01621459.2012.737745 |
| `matteson_james_2014` | Matteson and James (2014), A Nonparametric Approach for Multiple Change Point Analysis of Multivariate Data | https://doi.org/10.1080/01621459.2013.849605 |
| `taylor_2021` | Taylor, Killick, Burr, and Rogerson (2021), Assessing Daily Patterns Using Home Activity Sensors and Within Period Changepoint Detection | https://doi.org/10.1111/rssc.12472 |
| `martinez_hernandez_killick_2024` | Martinez-Hernandez and Killick (2024), Changepoint Detection on Daily Home Activity Pattern: A Sliced Poisson Process Method | https://doi.org/10.1093/biomtc/ujae114 |
| `harchaoui_bach_2008` | Harchaoui and Bach (2008), Kernel Change-point Analysis | https://proceedings.neurips.cc/paper/2008/hash/08b255a5d42b89b0585260b6f2360bdd-Abstract.html |
| `arlot_celisse_harchaoui_2019` | Arlot, Celisse, and Harchaoui (2019), A Kernel Multiple Change-point Algorithm via Model Selection | https://jmlr.org/papers/v20/16-155.html |
| `rahimi_recht_2007` | Rahimi and Recht (2007), Random Features for Large-Scale Kernel Machines | https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines |
| `yu_2010` | Yu (2010), Hidden Semi-Markov Models | https://doi.org/10.1016/j.artint.2009.11.011 |
| `manouchehri_bouguila_2023` | Manouchehri and Bouguila (2023), Human Activity Recognition with an HMM-Based Generative Model | https://doi.org/10.3390/s23031390 |

## Verification Meaning

- `verified`: has independent or paper-derived oracle tests.
- `partially_verified`: has focused unit or integration tests but lacks full independent oracles.
- `characterized_only`: current behavior is exercised but scientific equivalence is not established.
- `unverified`: implementation exists but important equations, proposals, gradients, or semantics remain unaudited.
- `not_implemented`: cited method is in scope but has no stable code path.
