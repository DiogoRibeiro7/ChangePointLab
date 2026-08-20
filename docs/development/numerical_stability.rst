Numerical Stability
===================

Numerical guards are part of the scientific contract. Algorithms should reject
invalid domains, raise on non-finite intermediate states, and use log-space or
centered calculations when those are mathematically equivalent to the original
objective.

Shared Policy
-------------

``changepoint_lab.core.numerics`` contains internal helpers for finite-value
checks, guarded exponentiation, and log-sum-exp. Overflow is not clipped into a
plausible value. Recoverable underflow may switch to an equivalent log-space
recursion and record diagnostics.

PELT
----

Gaussian segment costs use prefix sums on data centered by the global sample
mean. Segment residual sums of squares are unchanged by this translation, but
large-offset cancellation is reduced. Unknown-variance costs retain an
explicit positive variance floor for constant segments.

BOCPD
-----

BOCPD normally updates run-length mass on the probability scale. If an extreme
predictive likelihood underflows the tracked mass, the same recurrence is
recomputed in log space using likelihood-provided log predictive probabilities.
The recovery increments ``normalization_issues_`` and keeps log evidence
finite when the analytical predictive probability is finite.

Sliced Poisson
--------------

Sliced Poisson segment fitting evaluates the Poisson objective only on observed
quadrature exposure. Candidate Newton steps that overflow the intensity have
infinite objective value and are rejected by the line search. Non-finite
gradients, Hessians, or Newton steps stop the segment fit with an explicit
diagnostic message.

Kernel CPD
----------

Exact RBF Gram construction rejects non-finite inputs, non-finite pairwise
distances, invalid bandwidths, and NaN kernel exponents. Kernel underflow to
zero is acceptable for very distant points, but NaN or infinite kernel entries
are not.

E-Divisive
----------

E-Divisive validates finite observations and p-values. Distance statistics are
allowed to be zero on constant data; they must not produce NaN p-values or
non-finite split diagnostics.

State-Space Methods
-------------------

HSMM, SD-HMM, and SD-HMM mixture recursions use guarded log-sum-exp so
all-impossible slices remain ``-inf`` rather than becoming ``NaN``. SD-HMM
wrappers reject all-zero compositional rows because they cannot be normalized
onto the simplex. Mirror-descent emission updates raise a numerical stability
error if the exponentiated step overflows.

Remaining Limits
----------------

The guards are not a proof of broad statistical calibration. They cover
representative extreme values, constant data, near-degenerate objectives, and
invalid domains. Wider stress testing over large synthetic grids belongs in the
benchmark harness rather than the regular unit suite.
