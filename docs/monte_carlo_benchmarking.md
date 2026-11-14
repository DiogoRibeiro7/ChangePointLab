# Monte Carlo detector benchmarking (experimental)

`benchmark_gaussian_detectors` generates independent synthetic replicates,
constructs fresh registered offline detectors for each replicate, and records
their tolerance-aware detection scores.

```python
from changepoint_lab import (
    DetectorRegistry,
    PELT,
    benchmark_gaussian_detectors,
)
from changepoint_lab.algorithms.optimization.pelt import NormalMeanKnownVar

registry = DetectorRegistry()
registry.register(
    "pelt",
    lambda: PELT(cost_fn=NormalMeanKnownVar(sigma2=1), penalty=8, min_seg_len=5),
)

result = benchmark_gaussian_detectors(
    registry,
    lengths=[100, 100],
    means=[0, 4],
    standard_deviations=[1, 1],
    repetitions=100,
    tolerance=5,
    seed=42,
)
for summary in result.summaries:
    print(summary.detector, summary.mean_f1, summary.se_f1)
```

The same generated observations are shared across detectors within each
replicate, while NumPy child random streams separate the replicates. The
registry constructs new detector instances on every run.

Each summary is a *macro average across replicates* of precision, recall and
F1, with a sample-based standard error for each metric. These are Monte Carlo
standard errors, **not** confidence intervals or uncertainty about the true
break locations. For a single replicate, the standard error is defined as
zero because it cannot be estimated from one observation.

`result.replicates` retains the individual detector results for diagnostics.
`seed` controls synthetic data generation only. Stochastic detectors must
receive controlled random states through their registered factories.

The benchmark assumes independent Gaussian samples within each segment.
Stress tests for autocorrelation, outliers, missingness and temporal drift
require separate generators and are not implied by these baseline results.
