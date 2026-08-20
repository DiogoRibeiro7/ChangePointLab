# Benchmark Smoke Artifact

This report is generated from committed JSON and CSV artifacts. It is a
small harness smoke check, not a performance ranking.

## Environment

- Commit: `1e8f6771b838f044043293e68bc9cd271e4df288`
- Dirty tree while generated: `True`
- Python: `3.13.5`
- NumPy: `2.2.6`
- Platform: `Windows-11-10.0.26200-SP0`
- Deterministic measurements: `True`

## Runtime And Memory

| Dataset | Detector | Elapsed ns | Peak memory bytes |
| --- | --- | ---: | ---: |
| `piecewise_gaussian_n32` | `pelt_gaussian` | 0 | 0 |
| `piecewise_gaussian_n32` | `kernel_exact` | 0 | 0 |
| `piecewise_gaussian_n32` | `kernel_rff` | 0 | 0 |
| `piecewise_gaussian_n32` | `edivisive` | 0 | 0 |
| `piecewise_gaussian_n64` | `pelt_gaussian` | 0 | 0 |
| `piecewise_gaussian_n64` | `kernel_exact` | 0 | 0 |
| `piecewise_gaussian_n64` | `kernel_rff` | 0 | 0 |
| `piecewise_gaussian_n64` | `edivisive` | 0 | 0 |

## Boundary Accuracy

| Dataset | Detector | F1 | Predicted | Truth |
| --- | --- | ---: | --- | --- |
| `piecewise_gaussian_n32` | `pelt_gaussian` | 1.000 | `(16,)` | `(16,)` |
| `piecewise_gaussian_n32` | `kernel_exact` | 1.000 | `(16,)` | `(16,)` |
| `piecewise_gaussian_n32` | `kernel_rff` | 1.000 | `(16,)` | `(16,)` |
| `piecewise_gaussian_n32` | `edivisive` | 1.000 | `(16,)` | `(16,)` |
| `piecewise_gaussian_n64` | `pelt_gaussian` | 1.000 | `(32,)` | `(32,)` |
| `piecewise_gaussian_n64` | `kernel_exact` | 1.000 | `(32,)` | `(32,)` |
| `piecewise_gaussian_n64` | `kernel_rff` | 1.000 | `(32,)` | `(32,)` |
| `piecewise_gaussian_n64` | `edivisive` | 1.000 | `(32,)` | `(32,)` |

## Approximation Error

| Dataset | Approximate | Baseline | Boundary F1 | Score relative error |
| --- | --- | --- | ---: | ---: |
| `piecewise_gaussian_n32` | `kernel_rff` | `kernel_exact` | 1.000 | 0.071972 |
| `piecewise_gaussian_n64` | `kernel_rff` | `kernel_exact` | 1.000 | 0.097376 |

## Reproduction

Smoke artifact:

```bash
python scripts/benchmark_harness.py --profile smoke --deterministic-measurements
```

Real local timing and memory:

```bash
python scripts/benchmark_harness.py --profile full
```
