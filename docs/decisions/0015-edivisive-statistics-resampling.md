# 0015. E-Divisive statistics and resampling validation

Date: 2026-07-23

## Status

Accepted

## Context

E-Divisive uses an energy-distance split statistic and recursive significance
testing. The implementation already exposed IID permutation and block-based
dependent-data resampling in the low-level function, but the wrapper did not
expose the same controls. The split scan also excluded the final admissible
split point `m - min_size`, which is a numerical behavior change when the best
split is near a segment end.

## Decision

- Keep the energy statistic normalization:
  `(n_left * n_right / n) * energy_distance(left, right)`.
- Correct admissible split masking so all splits in
  `[min_size, m - min_size]` are considered.
- Preserve deterministic tie handling: `np.argmax` selects the first
  admissible maximizer.
- Expose significance, recursion, resampling, chunking, memmap, progress, and
  sequential execution controls through `EDivisive` as well as the low-level
  `edivisive(...)` function.
- Keep execution sequential. `n_jobs` is accepted only as `1`; other values are
  rejected to avoid undefined random-stream ordering.
- Treat block permutation and circular block bootstrap as package extensions
  for dependent data, not as Matteson-James paper parity claims.

## Consequences

Existing default wrapper calls remain IID and sequential. Results may differ
from previous versions when the best split is exactly `m - min_size`, because
that boundary is now correctly eligible. Significance has deterministic
empirical smoke-test evidence on small IID null fixtures; broader statistical
calibration and external implementation parity remain future work.
