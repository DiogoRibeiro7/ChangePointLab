Within-Period RJMCMC
====================

The within-period sampler follows Taylor, Killick, Burr, and Rogerson
(2021; DOI ``10.1111/rssc.12472``) as an adaptation for binary periodic
data.

State representation
--------------------

The one-segment model is represented as ``tau=()``. Non-empty states store
one periodic bin-end boundary per circular segment. A two-segment model
therefore has two boundaries, and singleton ``tau`` values are invalid.

Posterior target
----------------

For a state with ``m`` segments, the implementation uses:

* a Beta-Bernoulli segment marginal likelihood with uniform ``Beta(1, 1)``
  segment probability priors;
* a Dirichlet-multinomial prior over excess segment lengths
  ``delta_i = length_i - l`` for ``m > 1``;
* a truncated Poisson prior over ``m`` on ``1..floor(N / l)`` including the
  ``m * log(pois_lambda) - log(m!)`` terms.

The uniform circular anchor is marginalized as a constant over all states and
is omitted from Metropolis-Hastings ratios.

Proposal kernel
---------------

The sampler enumerates every feasible non-stay outcome for move, birth, and
death proposal families. Configured proposal weights must be positive and sum
to one. At a given state, impossible families are removed and available family
weights are renormalized exactly.

Reverse probabilities are computed by summing the enumerated proposal paths
from the proposed state back to the current state. The implementation does not
use nearest-boundary matching, average target counts, or random reverse
segment approximations.

Diagnostics and verification
----------------------------

``MCMCResult`` exposes ``acceptance_rate`` and ``move_counts``. Unit tests
cover tiny complete-state detailed balance, empirical stationary frequencies,
the non-unit ``pois_lambda`` posterior effect, and rotation invariance.
