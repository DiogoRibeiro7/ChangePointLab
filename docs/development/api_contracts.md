# Public API Result Contracts

ChangePointLab exposes estimator-style wrappers for several different
scientific workflows. A single "fit a matrix, return changepoints" contract is
not sufficient for all of them, so public wrappers use typed result objects with
shared fields only where the semantics match.

## Use-case interfaces

| Use case | Public shape | Result object | Notes |
| --- | --- | --- | --- |
| Offline segmentation | `fit`, `predict`, `fit_predict` | `SegmentationResult` | PELT, E-Divisive, and KernelCPD return right-exclusive changepoint boundaries, optional objective score, segment labels, and segment costs when available. |
| Online probability tracing | `run`; wrapper `fit_predict` for event extraction | `BOCPDResult` from `run`, `OnlineProbabilityResult` from wrapper prediction | BOCPD probabilities, MAP run lengths, and predictive means remain typed fields. Wrapper event extraction still uses the existing `cp_prob > 0.5` compatibility rule pending the BOCPD correctness workstream. |
| Posterior sampling | `fit`, `predict`, `fit_predict`, `result` | `PosteriorSampleResult` from wrapper prediction; `MCMCResult` from `.result` | Within-period inference exposes mode changepoints, posterior samples, changepoint histograms, and log posterior traces. |
| Latent-state decoding | `decode_viterbi`; wrapper `fit_predict` | `LatentStateResult` | HSMM and SD-HMM wrappers expose decoded states and changepoints derived from state or segment-duration boundaries. |
| Model selection | low-level selection functions | `ModelSelectionResult` for stable wrappers when added | Existing low-level KCP model-selection results remain available; a public wrapper should return selected model, criterion values, and selected changepoints. |

## Common result semantics

- `indices` are changepoint boundaries or event indices, described by
  `boundary_convention`.
- `right_exclusive` means each changepoint is the first index of the following
  segment.
- `time_index` means the index labels the observation time at which an online
  event probability crossed the wrapper's extraction rule.
- `score` is only set when a single comparable objective value exists.
- `objective_orientation` is only set when lower or higher scores have a stable
  meaning.
- `metadata` remains an extension channel for compatibility and non-core
  details, but essential public outputs are also available as typed fields.

## Compatibility notes

Existing callers that read `result.metadata["labels"]`, BOCPD probability
metadata, or HSMM decoded-state metadata continue to work. New code should prefer
typed fields such as `labels`, `costs_per_segment`, `cp_prob`, `states`, and
`segment_durations`.

Prompt 06 intentionally fixes wrappers that could not complete or that
misinterpreted already-typed core outputs:

- `KernelCPD.fit_predict` now builds a kernel prefix from the Gram matrix before
  calling the core KCP routine. The low-level KCP terminal-boundary bug remains
  recorded for the index-semantics workstream.
- `HSMM.fit_predict` now derives changepoints from nonzero sparse duration-end
  indicators.
- `SDHMMMixVI.fit_predict` now updates VI parameters without assigning into an
  immutable tuple.

