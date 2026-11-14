# Explicit detector factory registry (experimental)

`DetectorRegistry` manages named zero-argument factories for offline detectors.
It creates new instances each time you compare them, making repeated
experiments easier to organise without reusing fitted model state.

```python
from changepoint_lab import DetectorRegistry, PELT, simulate_gaussian_segments
from changepoint_lab.algorithms.optimization.pelt import NormalMeanKnownVar

registry = DetectorRegistry()
registry.register(
    "pelt-low-penalty",
    lambda: PELT(cost_fn=NormalMeanKnownVar(sigma2=1), penalty=4, min_seg_len=5),
)
registry.register(
    "pelt-high-penalty",
    lambda: PELT(cost_fn=NormalMeanKnownVar(sigma2=1), penalty=12, min_seg_len=5),
)

sample = simulate_gaussian_segments([100, 100], [0, 4], [1, 1], seed=7)
comparisons = registry.compare(sample.values, actual=sample.changepoints, tolerance=5)
for comparison in comparisons:
    print(comparison.name, comparison.scores.f1)
```

Use `registry.names` to discover registered identifiers and
`registry.build(["pelt-high-penalty"])` to construct a chosen subset. A
duplicate registration, unknown name or repeated selection raises an error.
Ordering follows registration order, or explicit selection order.

Factories should construct fresh objects and capture all required parameters;
the registry cannot guarantee independence if a factory deliberately returns
the same mutable object. No plugin scanning, implicit imports or global
registry are performed. Each registry is local to its caller.

The registry delegates evaluation to `compare_detectors`, including its
right-exclusive boundary requirement and one-to-one tolerance-based scoring.
