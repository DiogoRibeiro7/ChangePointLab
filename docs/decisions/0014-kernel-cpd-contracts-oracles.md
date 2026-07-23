# 0014. Kernel CPD contracts and oracles

Date: 2026-07-23

## Status

Accepted

## Context

`KernelCPD` needs to support kernel callables that either return a dense Gram
matrix directly or return the matrix with bandwidth metadata. Exact dense KCP
also has failure modes that should be rejected before dynamic programming:
non-finite values, asymmetric matrices, negative diagonal entries, indefinite
Gram matrices on small problems, and excessive dense-memory allocation.

The RFF implementation is an approximation to the RBF kernel path. It needs to
remain visibly distinct from exact dense KCP and retain the sampled bandwidth,
feature count, and seed in result metadata.

## Decision

- Keep the existing tuple return contract for `gram_rbf`, but add
  `KernelMatrix` for typed Gram-plus-metadata outputs.
- Validate dense Gram matrices inside `build_kernel_prefix`, including a
  configurable memory guard and small-matrix PSD tolerance.
- Let `KernelCPD` expose `min_size`, `method`, `grid_jump`, `max_seg_len`,
  `bandwidth`, exact/RFF mode, `RFFConfig`, PSD tolerance, and dense Gram memory
  guard as constructor configuration.
- Treat `KernelCPD(..., rff_config=...)` as an RFF run and include
  approximation metadata in the returned `SegmentationResult`.
- Validate exact KCP against independent feature-space and brute-force Gram
  objectives, and validate RFF on a deterministic high-feature convergence
  fixture.

## Consequences

Existing plain-matrix and `(K, gamma)` kernel callables continue to work.
Result metadata now has additional keys describing the kernel approximation and
solver configuration. Exact dense KCP remains quadratic in Gram storage; RFF
avoids dense Gram storage but is an approximation and still uses DP over
candidate segment endpoints.
