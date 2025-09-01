# Nonparametric Methods: E-Divisive

Nonparametric techniques avoid distributional assumptions, enabling flexible detection in complex datasets.

## Energy Statistics
E-Divisive uses energy statistics to quantify differences between segments. For segments $A$ and $B$:
$$\mathcal{E}(A,B) = \frac{2}{|A||B|} \sum_{a\in A} \sum_{b\in B} \|a-b\| - \frac{1}{|A|^2} \sum_{a,a'\in A} \|a-a'\| - \frac{1}{|B|^2} \sum_{b,b'\in B} \|b-b'\|.$$

## Permutation Testing
Significance is assessed via permutation: shuffle data many times to build a null distribution. Larger numbers of permutations improve accuracy but increase runtime.

## Distance Metrics
- Euclidean distance is common; alternatives (Manhattan, cosine) can emphasize different features.
- Exponent parameters can accentuate large deviations.

## Multivariate Data
E-Divisive handles multivariate sequences by computing distances in the multivariate space, making it robust to correlated changes.

See [E-Divisive API](../api/edivisive.rst) for further details.
