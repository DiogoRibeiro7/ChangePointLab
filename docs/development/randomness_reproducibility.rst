Randomness and Reproducibility
==============================

ChangePointLab production algorithms must not mutate process-global random
state. Stochastic methods use owned ``numpy.random.Generator`` instances
created from explicit seeds or caller-supplied generators.

Public result objects include reproducibility provenance for stochastic paths
where the current public wrapper returns a typed result. The provenance records
the seed, generator family, and method-specific stochastic settings such as
iteration counts, resampling policy, or chain configuration.

Seeded execution
----------------

Passing the same seed to a stochastic method is expected to replay the same
sequence of random draws for that method and configuration. This is a
per-method contract: changing iteration counts, resampling policy, candidate
generation, or proposal logic can intentionally change the trace and should be
recorded as a behavior change.

Caller-supplied generators
--------------------------

When a method accepts a ``Generator``, the generator is treated as stateful.
Repeated calls with the same generator object continue from the consumed state;
repeated calls with freshly constructed generators from the same seed replay.

Parallel or multi-stream execution
----------------------------------

Parallel chains, swaps, bootstrap loops, and related independent stochastic
subtasks should use child streams created through ``SeedSequence.spawn``. This
keeps sequential and parallel execution deterministic for a seed while avoiding
accidental reuse of the same stream for logically independent tasks.

Compatibility notes
-------------------

The migration from module-level ``np.random.seed`` and Python ``random`` to
``Generator`` changes some legacy seeded traces. The baseline fixture records
the current ``Generator`` outputs for compatibility, but the within-period
sampler remains under scientific audit for proposal accounting and posterior
target validation.
