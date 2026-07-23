Within-Period Reproduction
==========================

The within-period reproduction artifacts are generated from a single code path:

.. code-block:: console

   poetry run python scripts/run_within_period_reproduction.py --profile ci --output artifacts/within_period_reproduction

The ``ci`` profile is a deterministic smoke profile with short chains. It is
intended to prove that the full workflow runs from source and that outputs are
well formed. The ``research`` profile uses longer chains and broader prior
sensitivity settings for interpretation:

.. code-block:: console

   poetry run python scripts/run_within_period_reproduction.py --profile research --output artifacts/within_period_reproduction

Scope separation
----------------

The output JSON separates:

* ``paper_consistent``: synthetic Bernoulli periodic scenarios based on the
  model and simulation structure described by Taylor, Killick, Burr, and
  Rogerson (2021);
* ``mysense_extension``: a generated 96-bin daily example with chair, doors,
  kettle, tap, and toilet sensor streams aggregated into an ``any_activity``
  binary series.

The proprietary passive-sensor records used in the paper are not bundled with
the package. Reproduction artifacts therefore report deterministic analogues
and discrepancies instead of claiming exact recreation of the paper figures.

Generated outputs
-----------------

The script writes:

* ``within_period_reproduction_summary.json`` with posterior summaries,
  diagnostics, prior sensitivity, and known discrepancies;
* ``paper_scenario_summary.csv`` for paper-style synthetic scenarios;
* ``prior_sensitivity.csv`` for selected prior settings;
* ``mysense_sensor_rates.csv`` for generated sensor activity rates;
* ``paper_changepoint_mass.svg`` for marginal changepoint mass by scenario.

Validation
----------

The test suite executes the ``ci`` profile, checks simulation coverage across
no-change, one-change, multiple-change, weak-signal, and boundary-crossing
cases, and executes the reproduction notebook without network access.
