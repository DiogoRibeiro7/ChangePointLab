# Baseline test truth

Captured on: `2026-08-19T15:43:57.234711+00:00`
Git commit: `bf3532814c1bb5c7598e4d68cadedfede8c22027`
Branch: `main`
Python: `3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]`
NumPy: `2.2.6`
Platform: `Windows-11-10.0.26200-SP0`

This audit records the current executable state before scientific behavior changes.
Passing checks do not upgrade any scientific verification status.

## Command Results

| Check | Result | Duration | Parsed Summary |
| --- | --- | ---: | --- |
| `unit_tests` | pass | 103.930s | `{"passed": 134, "warnings": 12}` |
| `integration_tests` | pass | 10.936s | `{"passed": 3, "warnings": 2}` |
| `slow_tests` | pass | 38.568s | `{"passed": 36, "warnings": 14}` |
| `benchmark_tests` | pass | 5.015s | `{"passed": 12, "warnings": 2}` |
| `ruff` | pass | 0.168s | `{}` |
| `mypy` | pass | 0.447s | `{}` |
| `pydocstyle` | pass | 3.051s | `{}` |
| `sphinx_docs` | pass | 1.481s | `{}` |
| `pdoc_docs` | pass | 1.937s | `{}` |
| `package_build` | pass | 3.927s | `{}` |
| `distribution_validation` | pass | 0.165s | `{}` |
| `installed_wheel_smoke` | pass | 16.773s | `{}` |
| `fresh_process_import` | pass | 0.389s | `{}` |
| `stable_top_level_exports` | pass | 0.423s | `{}` |
| `tiny_public_executions` | pass | 0.400s | `{}` |
| `cpd_help` | pass | 0.461s | `{}` |
| `bocpd_help` | pass | 0.494s | `{}` |
| `within_period_help` | pass | 0.482s | `{}` |
| `cpd_tiny_execution` | pass | 1.387s | `{}` |
| `bocpd_tiny_execution` | pass | 1.647s | `{}` |
| `within_period_tiny_execution` | fail (1) | 0.419s | `{"contains_error_text": true}` |

## Failure Characterization

### `within_period_tiny_execution`

```text
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\diogo\work_code\ChangePointLab\src\changepoint_lab\algorithms\bayesian\within_period\cli.py", line 334, in <module>
    main()
    ~~~~^^
  File "C:\Users\diogo\work_code\ChangePointLab\src\changepoint_lab\algorithms\bayesian\within_period\cli.py", line 294, in main
    samples_tau, cp_hist, mode_tau, log_posts = _fit_rjmcmc(
                                                ~~~~~~~~~~~^
        x=x, prior=prior, iters=args.iters, burn=args.burn, thin=args.thin, seed=args.seed
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\diogo\work_code\ChangePointLab\src\changepoint_lab\algorithms\bayesian\within_period\cli.py", line 162, in _fit_rjmcmc
    return result.samples_tau, result.changepoint_hist, result.mode_tau, result.log_posteriors
           ^^^^^^^^^^^^^^^^^^
  File "C:\Users\diogo\work_code\ChangePointLab\src\changepoint_lab\algorithms\bayesian\within_period\__init__.py", line 93, in __getattr__
    raise AttributeError(name)
AttributeError: samples_tau
```


## Remaining Scientific Limitations

- This change records executable evidence only.
- Existing partial or unverified methods remain at their current verification status.
- Follow-up changes should address correctness, API, documentation, and release-readiness items separately.
