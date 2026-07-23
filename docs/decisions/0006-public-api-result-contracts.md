# Decision 0006: Public API Result Contracts

Date: 2026-07-23

## Status

Accepted.

## Context

ChangePointLab combines offline segmentation, online Bayesian probabilities,
posterior sampling, and latent-state decoding. The old public surface forced
most wrappers into `ChangePointResult` plus dictionary metadata, which meant
essential outputs such as labels, run-length probabilities, sampled
changepoints, and decoded states had no stable typed location.

The audit also recorded three public wrapper defects:

- `KernelCPD.fit_predict` crashed on its default RBF kernel because the wrapper
  passed `(K, gamma)` directly to a core function expecting `KernelPrefix`.
- `HSMM.fit_predict` converted sparse duration-end indicators with
  `np.cumsum`, producing duplicate and zero-derived changepoints.
- `SDHMMMixVI.fit_predict` assigned into immutable parameter tuples.

## Decision

Keep `ChangePointResult` as the common compatibility base and introduce typed
specializations:

- `SegmentationResult`
- `OnlineProbabilityResult`
- `PosteriorSampleResult`
- `LatentStateResult`
- `ModelSelectionResult`

Wrappers may return a subclass of `ChangePointResult` when algorithm-specific
fields are essential. Compatibility `metadata` keys remain available, but new
code should use typed fields.

Distinct protocols document the main public interaction shapes:
`OfflineDetector`, `OnlineDetector`, `PosteriorSampler`, and
`LatentStateDecoder`.

## Consequences

- Existing callers expecting `ChangePointResult` continue to receive an
  instance of that base class.
- Type-aware callers can now rely on explicit fields for labels, costs,
  probabilities, posterior samples, and latent states.
- Three exported wrapper paths now execute on minimal examples.
- KCP low-level terminal-boundary semantics and BOCPD probability calibration
  remain out of scope for this decision and stay in the risk register.

