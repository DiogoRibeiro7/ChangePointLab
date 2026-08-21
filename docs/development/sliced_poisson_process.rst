Sliced Poisson Process
======================

The sliced Poisson process detector implements a local adaptation of
Martinez-Hernandez and Killick (2024; DOI ``10.1093/biomtc/ujae114``).
Each repeated period, such as one day, is treated as one observation from an
inhomogeneous Poisson process on ``[0, period)``. Changepoints are detected
across the sequence of periods.

Faithful baseline
-----------------

The baseline API is ``SlicedPoissonCPD``. It accepts a sequence of event-time
periods. A period can be either a plain sequence of event times or an
``EventPeriod`` with explicit exposure intervals. The fitted log-intensity in
each segment is represented by an open uniform B-spline basis:

.. code-block:: python

   from changepoint_lab import SlicedPoissonCPD, SlicedPoissonConfig

   periods = [(0.2, 0.4), (0.3,), (3.0, 3.2), (3.1,)]
   cfg = SlicedPoissonConfig(period=24.0, n_basis=5, degree=3)
   result = SlicedPoissonCPD(cfg).fit_predict(periods)

The segment cost is minus twice the optimized inhomogeneous Poisson
log-likelihood. This additive objective is passed through the shared PELT
interface. The optimizer currently uses exact candidate retention for bundled
costs, so ``K`` is retained only as a compatibility argument.

Point-process sufficient statistics
-----------------------------------

The likelihood uses B-spline basis sums at event times for the log-intensity
term. Segment event totals, diagnostics, and zero-event handling use a separate
integer prefix count over periods. This keeps total event counts exact even at
spline knot boundaries, with repeated event times, or with large event totals.
Development assertions compare the partition-of-unity basis sums against the
direct counts so basis regressions fail early.

Exposure intervals
------------------

Observed intervals are supplied per period:

.. code-block:: python

   from changepoint_lab import EventPeriod

   period = EventPeriod(event_times=(8.25, 9.0), exposure_intervals=((6.0, 12.0),))

Exposure intervals use half-open ``[start, end)`` semantics on
``[0, period)``. Starts are included, ends are excluded, and an event exactly
at ``period`` is invalid. Events outside observed exposure intervals raise
``ValueError``.

Intervals are normalized before fitting. Unordered, overlapping, nested,
duplicate, and touching intervals are converted to their sorted
non-overlapping union. For example, ``((0.0, 0.5), (0.25, 0.75),
(0.75, 1.0))`` is treated as ``((0.0, 1.0),)``. This preserves legacy
non-overlapping inputs while preventing duplicated observed time from being
counted more than once.

Likelihood integrals use deterministic interval-local Gauss-Legendre
quadrature over each canonical observed interval. ``quadrature_points`` is the
number of nodes per interval, not a global period-wide mask. This means a valid
narrow observation window always contributes positive numerical exposure even
when it is much shorter than the display grid spacing. Constant-intensity
models integrate to the analytical observed-time measure. For non-constant
B-spline intensities, increase ``quadrature_points`` to reduce approximation
error; ``SlicedPoissonResult.diagnostics["exposure_integration"]`` records the
scheme, nodes per interval, total nodes, and error-control note.

Marked extension
----------------

Marked sensor streams are intentionally outside the faithful unmarked baseline.
``fit_marked_sliced_poisson(..., mode="independent")`` fits one independent
detector per mark. ``mode="shared_baseline"`` raises ``NotImplementedError``;
it is not approximated silently.

Diagnostics
-----------

``SlicedPoissonResult`` exposes segment fits, fitted intensity grids, costs,
labels, optimization convergence messages, and the generic
``SegmentationResult`` view through ``to_changepoint_result()``.

Optimizer failure policy
------------------------

Segment fits use a stabilized Newton objective where the objective, gradient,
and Hessian are evaluated from the same finite-checked intensity values. Each
iteration rejects non-finite objectives, gradients, Hessians, or candidate
steps.

``SlicedPoissonConfig.optimizer_failure_policy`` controls what happens when a
segment cannot be fitted:

``"raise"``
   Default. Raise ``NumericalStabilityError`` before the invalid segment cost
   can affect PELT selection.

``"retry"``
   Retry once from a zero-weight initialization with the same deterministic
   damping and fallback rules. If the retry still fails, return an infinite
   segment cost.

``"penalize_invalid"``
   Return an infinite segment cost without raising. This keeps the segment
   ineligible for optimum selection while preserving diagnostics.

Every failed non-raising fit reports its convergence reason, accepted step
scale, Hessian condition estimate, and retry count in the stored
``SegmentFit``.

Limitations
-----------

The implementation uses a NumPy-only damped Newton solver and fixed-order
Gauss-Legendre quadrature. The quadrature is deterministic but not adaptive.
The Howz data from the paper are not bundled, so tests use analytical cases
and deterministic simulations rather than paper-data parity.
