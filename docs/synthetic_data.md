# Synthetic changepoint datasets (experimental)

Use `simulate_gaussian_segments` to create independent Gaussian time series
with known right-exclusive changepoint boundaries. The generator supports mean
shifts, variance shifts, combinations of both, and constant segments.

```python
from changepoint_lab import score_changepoints, simulate_gaussian_segments

sample = simulate_gaussian_segments(
    lengths=[100, 120, 80],
    means=[0.0, 2.0, 2.0],
    standard_deviations=[1.0, 1.0, 3.0],
    seed=42,
)

# Ground truth: [100, 220] (observation-index, right-exclusive boundaries).
assert sample.changepoints.tolist() == [100, 220]
assert sample.values.shape == (300,)

# Replace this example with boundaries returned by a fitted detector.
predicted = [101, 218]
scores = score_changepoints(
    predicted, sample.changepoints, n_samples=len(sample.values), tolerance=3
)
assert scores.f1 == 1.0
```

All three parameter sequences must have equal nonzero length. `lengths` must
contain positive integers; means must be finite, while standard deviations
must be finite and nonnegative. A standard deviation of zero yields constant
samples. A single segment has no changepoints.

An explicit seed produces the same observations across repeated calls under
the same NumPy random-generator implementation. For controlled Monte Carlo
experiments, supply a caller-owned `numpy.random.Generator` instead of a seed.

For autocorrelated observations, use `simulate_gaussian_ar1_segments`:

```python
from changepoint_lab import simulate_gaussian_ar1_segments

sample = simulate_gaussian_ar1_segments(
    lengths=[200, 200],
    means=[0.0, 3.0],
    standard_deviations=[1.0, 1.5],
    autocorrelation=0.7,
    seed=42,
)
assert sample.changepoints.tolist() == [200]
```

This generator uses a stationary Gaussian AR(1) process within each segment.
The supplied standard deviations describe the **marginal distribution**, not
the innovations; the innovation standard deviation is multiplied by
`sqrt(1 - autocorrelation**2)`. The process restarts from a stationary draw at
each boundary. Consequently, changepoints are known and do not depend on a
previous segment's latent state. Autocorrelation must be strictly between -1
and 1. Setting it to zero reproduces `simulate_gaussian_segments` exactly for
the same seed.

These datasets provide ground truth for benchmark correctness but do not yet
model gradual drift, outliers or missingness. Those remain separate stress-test
settings, not implicit assumptions of the baseline generators.
