# 0009 - Within-period reproduction scope

Date: 2026-07-23

## Status

Accepted

## Context

Taylor, Killick, Burr, and Rogerson (2021) define a within-period binary
changepoint model for passive home-activity sensors and report simulation and
case-study results. The public article text describes the model, simulation
structure, 96 daily bins for 15-minute data, one-hour minimum segment lengths,
and qualitative case-study findings. The proprietary individual-level sensor
records used in the case study are not bundled with this repository.

## Decision

Provide a reproducible local workflow that separates:

- paper-consistent synthetic Bernoulli periodic scenarios;
- a generated MySense-style multi-sensor extension;
- explicit discrepancy notes for unavailable data and indexing conversions.

The CI profile is a deterministic execution check. The research profile is the
intended setting for interpretable posterior summaries.

## Consequences

The repository can now generate tables and figures from source with one command
without network access. It does not claim exact recreation of the paper's
proprietary case-study data or the full 1000-replication simulation tables.
