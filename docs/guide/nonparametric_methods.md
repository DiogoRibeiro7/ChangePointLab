# Nonparametric Methods: E-Divisive

Nonparametric techniques avoid distributional assumptions, enabling flexible detection in complex datasets.

## Energy Statistics
E-Divisive uses energy statistics to quantify differences between segments. For segments $A$ and $B$:
$$\mathcal{E}(A,B) = \frac{2}{|A||B|} \sum_{a\in A} \sum_{b\in B} \|a-b\| - \frac{1}{|A|^2} \sum_{a,a'\in A} \|a-a'\| - \frac{1}{|B|^2} \sum_{b,b'\in B} \|b-b'\|.$$

## Permutation Testing
Significance is assessed by comparing the observed maximum split statistic for a tested segment against a resampled null distribution. For IID data, use `resample="iid"` to shuffle observations. For dependent data, `resample="block-permutation"` and `resample="circular-block-bootstrap"` are package extensions that preserve short-range dependence inside blocks; they are validated on deterministic synthetic fixtures but should not be treated as paper-parity defaults.

## Distance Metrics
- Euclidean distance is common; alternatives (Manhattan, cosine) can emphasize different features.
- Exponent parameters can accentuate large deviations.

## Multivariate Data
E-Divisive handles multivariate sequences by computing distances in the multivariate space, making it robust to correlated changes.

## Execution Semantics
Recursive splitting uses a deterministic breadth-first segment queue. Progress output follows that queue order. Parallel execution is not implemented; `n_jobs` is exposed and must be `1` so random-stream ordering remains explicit.

See [E-Divisive API](../api/edivisive.rst) for further details.
