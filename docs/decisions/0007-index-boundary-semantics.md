# Decision 0007: Index and Boundary Semantics

Date: 2026-07-23

## Context

The package exposes algorithms that report linear segment boundaries, online
event times, latent-state transitions, sparse duration ends, and within-period
circular changepoints. Treating all of these as plain integer arrays made
off-by-one errors easy to miss. The KCP implementation demonstrated this risk:
backtracking retained terminal endpoint `n` as a changepoint and dropped the
interior boundary selected by the dynamic program.

## Decision

Use `right_exclusive` as the canonical convention for linear segmentation
results. A changepoint `k` means the new segment starts at observation `k`; the
corresponding segment edges are `[0, *changepoints, n]`.

Keep online BOCPD event extraction as `time_index`, because the wrapper reports
times where the compatibility threshold rule fires rather than a proven
offline segmentation.

Represent circular within-period changepoints with an explicit
`periodic_bin_end` convention and `CircularChangePoints`. A circular
changepoint is the final bin of a period segment, so terminal linear boundary
logic is not applied.

## Consequences

- Linear endpoints `0` and `n`, duplicate changepoints, unsorted changepoints,
  and minimum-segment violations are validation errors.
- Shared helpers convert among changepoints, edges, labels, and slices.
- Exact and RFF KCP now return the interior boundary `[2]` on the four-point
  kernel oracle, with edges `[0, 2, 4]`.
- Within-period wrapper results advertise `periodic_bin_end` so downstream code
  does not silently treat circular bin ends as linear right-exclusive edges.
