# Comparing offline changepoint detectors (experimental)

`compare_detectors` applies named offline detectors to the same observations
and optionally scores their detected boundaries using shared reference events.

```python
from changepoint_lab import PELT, compare_detectors, simulate_gaussian_segments
from changepoint_lab.algorithms.optimization.pelt import NormalMeanKnownVar

sample = simulate_gaussian_segments([100, 100], [0, 5], [1, 1], seed=7)
detectors = {
    "pelt": PELT(cost_fn=NormalMeanKnownVar(sigma2=1), penalty=8, min_seg_len=5),
}
comparisons = compare_detectors(
    detectors, sample.values, actual=sample.changepoints, tolerance=5
)
for entry in comparisons:
    print(entry.name, entry.result.indices.tolist(), entry.scores.f1)
```

The result is a tuple of `DetectorComparison` records in input mapping order.
Each record contains the detector's original `ChangePointResult` and optional
`ChangePointScores`. Without `actual`, scores are `None`.

The helper requires the offline `fit_predict(x)` interface and a typed result
with `right_exclusive` boundaries. It rejects online probability indices and
periodic bin-end boundaries rather than silently mixing incompatible
conventions. Every detector gets its own observation copy. Detector instances
are fitted in place; construct new instances for each independent replicate.

Detection precision/recall/F1 can be compared, but raw algorithm objective
scores must not be interpreted on a common scale. Computational timing,
parameter tuning, uncertainty calibration and a detector factory registry are
separate benchmark concerns, not implicit features of this entry point.
