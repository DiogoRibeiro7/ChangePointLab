# Decision 0008: Randomness and Reproducibility

Date: 2026-07-23

## Context

Several stochastic paths used process-global random state through
`np.random.seed`, `np.random.*`, or Python's module-level `random`. That made
composition fragile: one algorithm could perturb another algorithm's draws, and
parallel chains could accidentally share the same stream semantics.

## Decision

Use `numpy.random.Generator` as the standard stochastic interface for
production algorithms. Existing public `seed` and `random_state` parameters now
construct local generators instead of mutating global state.

Use spawned child streams for logically independent streams, such as
parallel-tempering cold, hot, and swap draws.

Expose stochastic provenance through typed result objects. The provenance
records the seed, generator family, and method-specific stochastic settings
needed to understand replay expectations.

## Consequences

- Stochastic methods with the same seed and configuration replay within the
  same implementation version.
- Caller-supplied `Generator` objects are stateful; reusing the same object
  consumes its stream.
- Some legacy seeded traces changed during the migration from module-level RNG
  state to `Generator`.
- This resolves the hidden-global-RNG risk but does not by itself verify the
  scientific posterior or proposal equations for within-period RJMCMC.
