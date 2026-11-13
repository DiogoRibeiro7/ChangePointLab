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

These datasets provide ground truth for benchmark correctness but do not model
autocorrelation, drift, outliers or missingness. Those are separate stress-test
settings, not implicit assumptions of this baseline generator.
