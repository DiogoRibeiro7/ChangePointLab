Metamorphic Properties
======================

Metamorphic tests encode transformations that should preserve a result because
of the model objective, not because of a single fixture's expected changepoint
locations. They are narrower than correctness proofs and should be tied to the
assumptions below.

Segmentation Conventions
------------------------

Linear changepoints are right-exclusive interior boundaries. For any valid
partition, converting changepoints to half-open edges and contiguous labels must
round-trip back to the same boundaries. This follows directly from the
``[start, stop)`` segment convention.

PELT
----

Gaussian PELT costs are translation invariant because centering by the segment
mean leaves residual sums of squares unchanged. Positive rescaling preserves
boundaries for the known-variance cost only when the variance parameter is
rescaled by the square of the same factor; otherwise the data units change
relative to the penalty and the selected segmentation may change.

The beta-binomial PELT cost with symmetric prior parameters is invariant under
binary label complement because successes and failures exchange roles.

BOCPD
-----

The Beta-Bernoulli BOCPD recursion is invariant under binary complement only
when the Beta prior is symmetric. With asymmetric prior parameters, zeros and
ones have different prior predictive probabilities, so complementing the stream
is intentionally not an invariance.

E-Divisive
----------

Energy-distance split statistics are invariant to translation and scale
homogeneously under positive rescaling. Permutation p-values compare statistics
on the same transformed distance scale, so detected boundaries are expected to
match under fixed seeds for the tested deterministic fixtures.

Kernel CPD
----------

RBF kernel segmentation depends on pairwise squared distances. Translation and
orthogonal rotation preserve those distances. With the median bandwidth
heuristic, positive rescaling multiplies all squared distances and the median by
the same factor, preserving the resulting Gram matrix.

Within-Period Detection
-----------------------

Within-period boundaries use circular ``periodic_bin_end`` semantics. Rotating
each repeated period and rotating the boundary set by the same offset preserves
segment lengths and the circular log posterior.

Sliced Poisson
--------------

The constant-intensity sliced Poisson basis has one segment-wide rate parameter.
Under that restricted model, within-period event phase does not affect segment
costs; only counts and exposure matter. This property must not be generalized
to spline bases that model timing shape.

State-Space Methods
-------------------

HSMM Viterbi decoding is invariant to adding any per-time constant to all state
log-likelihoods because every candidate state path receives the same additive
observation term at that time.

Scaled-Dirichlet HMM inputs are proportional feature vectors normalized row by
row onto the simplex. Multiplying each row by a positive scalar preserves the
normalized composition and should preserve seeded wrapper outputs.
