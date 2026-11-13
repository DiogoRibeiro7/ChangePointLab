# Changepoint detection scores (experimental)

`score_changepoints` evaluates a detector's predicted boundaries against a
reference annotation using precision, recall and F1.

```python
from changepoint_lab import score_changepoints

scores = score_changepoints(
    predicted=[101, 201, 301],
    actual=[100, 205],
    n_samples=400,
    tolerance=5,
)
assert scores.true_positives == 2
assert scores.false_positives == 1
assert scores.false_negatives == 0
```

Both inputs must contain **strictly increasing, unique, integer** observation
indices. Index `k` denotes a right-exclusive segment boundary between samples
`k - 1` and `k`, so only `1..n_samples-1` is valid. Online probability indices
and circular/periodic bin-end changepoints require an explicit conversion before
using these scores.

Every predicted boundary can be matched to at most one reference boundary, and
vice versa. A match requires `abs(predicted - actual) <= tolerance`. The
deterministic left-to-right matching algorithm maximises match count; it does
not guarantee minimum total temporal error among maximum-cardinality matchings.
Unmatched predictions are false positives and unmatched reference boundaries
are false negatives. Precision, recall and F1 are all `1.0` when both sets are
empty; when exactly one set is empty all three scores are `0.0`.

The `matched_pairs` field contains `(predicted, actual)` index pairs and can be
used to inspect temporal offsets without interpreting the F1 score as a measure
of boundary localisation error.
