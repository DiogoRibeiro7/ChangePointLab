from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from ....core.random import make_rng, spawn_rngs
from .within_period_cpd import MCMCResult, ModelPrior, RJConfig, Tau, WithinPeriodCore, _is_valid_tau


@dataclass(frozen=True)
class PeriodicBernoulliScenario:
    """Synthetic periodic binary scenario for within-period replication."""

    name: str
    source_scope: str
    period: int
    days: int
    min_segment_length: int
    boundaries: Tau
    probabilities: tuple[float, ...]
    description: str


@dataclass(frozen=True)
class ReplicationProfile:
    """Runtime profile for deterministic reproduction scripts."""

    name: str
    iters: int
    burn: int
    thin: int
    seed: int
    exact_state_limit: int
    prior_lambdas: tuple[float, ...]


@dataclass(frozen=True)
class SensorSeries:
    """Synthetic MySense-style multi-sensor binary data."""

    period: int
    days: int
    sensors: Mapping[str, NDArray[np.bool_]]
    any_activity: NDArray[np.bool_]
    source_scope: str
    description: str


@dataclass(frozen=True)
class FitSummary:
    """Compact fit summary for a single within-period reproduction run."""

    scenario: str
    source_scope: str
    period: int
    days: int
    min_segment_length: int
    true_boundaries: Tau
    mode_tau: Tau
    true_tau_probability: float
    hpcs_size_95: int
    acceptance_rate: float
    sample_count: int


CI_PROFILE = ReplicationProfile(
    name="ci",
    iters=120,
    burn=40,
    thin=10,
    seed=20260723,
    exact_state_limit=5000,
    prior_lambdas=(0.5, 1.0, 2.0),
)

RESEARCH_PROFILE = ReplicationProfile(
    name="research",
    iters=30_000,
    burn=10_000,
    thin=10,
    seed=20260723,
    exact_state_limit=200_000,
    prior_lambdas=(0.5, 1.0, 2.0, 4.0),
)


def profile_by_name(name: str) -> ReplicationProfile:
    """Return a named deterministic reproduction profile."""
    profiles = {"ci": CI_PROFILE, "research": RESEARCH_PROFILE}
    try:
        return profiles[name]
    except KeyError as exc:
        raise ValueError(f"unknown replication profile: {name}") from exc


def probability_grid(period: int, boundaries: Tau, probabilities: tuple[float, ...]) -> NDArray[np.float64]:
    """Construct a periodic probability grid from circular bin-end boundaries."""
    if boundaries and len(boundaries) != len(probabilities):
        raise ValueError("non-empty boundaries require one probability per segment.")
    if not boundaries and len(probabilities) != 1:
        raise ValueError("the one-segment model requires exactly one probability.")
    if any(p < 0.0 or p > 1.0 for p in probabilities):
        raise ValueError("probabilities must be in [0, 1].")

    grid = np.empty(period, dtype=float)
    if not boundaries:
        grid[:] = probabilities[0]
        return grid

    previous = boundaries[-1]
    for idx, boundary in enumerate(boundaries):
        length = (boundary - previous) % period
        length = period if length == 0 else length
        start = (previous + 1) % period
        for offset in range(length):
            grid[(start + offset) % period] = probabilities[idx]
        previous = boundary
    return grid


def simulate_periodic_bernoulli(
    scenario: PeriodicBernoulliScenario,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> NDArray[np.bool_]:
    """Simulate repeated periodic Bernoulli observations for a scenario."""
    rng = make_rng(seed=seed, rng=rng)
    grid = probability_grid(scenario.period, scenario.boundaries, scenario.probabilities)
    draws = rng.binomial(1, np.tile(grid, scenario.days))
    return draws.astype(bool)


def paper_replication_scenarios() -> tuple[PeriodicBernoulliScenario, ...]:
    """
    Return paper-style synthetic scenarios with documented local indexing.

    Taylor et al. report event positions using one-based period-end notation in
    the accessible article text. These scenarios use the package convention:
    zero-based periodic bin-end indices.
    """
    return (
        PeriodicBernoulliScenario(
            name="paper_monte_carlo_n24",
            source_scope="paper_consistent",
            period=24,
            days=30,
            min_segment_length=4,
            boundaries=(7, 15, 23),
            probabilities=(0.25, 0.50, 0.60),
            description="Section 4.1 analogue: N=24, l=4, three evenly spaced segments.",
        ),
        PeriodicBernoulliScenario(
            name="no_change",
            source_scope="paper_consistent",
            period=96,
            days=35,
            min_segment_length=4,
            boundaries=(),
            probabilities=(0.50,),
            description="Section 4.2 d=0 analogue with no changepoint signal.",
        ),
        PeriodicBernoulliScenario(
            name="one_activity_window",
            source_scope="paper_consistent",
            period=96,
            days=35,
            min_segment_length=4,
            boundaries=(31, 79),
            probabilities=(0.70, 0.30),
            description="Two-segment wake/sleep-style case with strong signal.",
        ),
        PeriodicBernoulliScenario(
            name="weak_signal",
            source_scope="paper_consistent",
            period=96,
            days=35,
            min_segment_length=4,
            boundaries=(31, 79),
            probabilities=(0.55, 0.45),
            description="Two-segment case matching the weak d=0.1 signal pattern.",
        ),
        PeriodicBernoulliScenario(
            name="multiple_activity_windows",
            source_scope="paper_consistent",
            period=96,
            days=35,
            min_segment_length=4,
            boundaries=(23, 39, 55, 83),
            probabilities=(0.25, 0.75, 0.45, 0.10),
            description="Multiple daily activity regimes.",
        ),
        PeriodicBernoulliScenario(
            name="boundary_crossing_sleep",
            source_scope="paper_consistent",
            period=96,
            days=35,
            min_segment_length=4,
            boundaries=(27, 87),
            probabilities=(0.80, 0.08),
            description="Sleep interval crosses midnight, exercising circular pooling.",
        ),
    )


def mysense_sensor_example(
    *,
    days: int = 56,
    period: int = 96,
    rng: np.random.Generator | None = None,
    seed: int | None = 715,
) -> SensorSeries:
    """Create a synthetic MySense-oriented five-sensor daily binary example."""
    rng = make_rng(seed=seed, rng=rng)
    sensor_windows = {
        "chair": ((31, 39, 55, 83), (0.04, 0.42, 0.20, 0.03)),
        "doors": ((27, 35, 71, 87), (0.05, 0.16, 0.10, 0.02)),
        "kettle": ((27, 35, 43, 51, 75, 83), (0.02, 0.30, 0.03, 0.22, 0.04, 0.01)),
        "tap": ((27, 39, 67, 83), (0.03, 0.24, 0.18, 0.02)),
        "toilet": ((23, 31, 83, 91), (0.03, 0.10, 0.04, 0.08)),
    }
    sensors: dict[str, NDArray[np.bool_]] = {}
    for name, (boundaries, probs) in sensor_windows.items():
        scenario = PeriodicBernoulliScenario(
            name=name,
            source_scope="mysense_extension",
            period=period,
            days=days,
            min_segment_length=4,
            boundaries=boundaries,
            probabilities=probs,
            description=f"Synthetic {name} activity profile.",
        )
        sensors[name] = simulate_periodic_bernoulli(scenario, rng=rng)

    stacked = np.vstack(list(sensors.values()))
    return SensorSeries(
        period=period,
        days=days,
        sensors=sensors,
        any_activity=np.any(stacked, axis=0),
        source_scope="mysense_extension",
        description="Synthetic MySense-style aggregate: any chair, door, kettle, tap, or toilet activity.",
    )


def enumerate_valid_taus(period: int, min_segment_length: int) -> tuple[Tau, ...]:
    """Enumerate valid circular tau states for small periods."""
    states: list[Tau] = [()]
    max_segments = period // min_segment_length
    for segment_count in range(2, max_segments + 1):
        states.extend(
            tau
            for tau in _combinations(range(period), segment_count)
            if _is_valid_tau(tau, period, min_segment_length)
        )
    return tuple(states)


def _combinations(values: Iterable[int], length: int) -> Iterable[Tau]:
    """Typed wrapper around itertools.combinations."""
    from itertools import combinations

    return (tuple(combo) for combo in combinations(values, length))


def fit_scenario(
    scenario: PeriodicBernoulliScenario,
    profile: ReplicationProfile,
    *,
    seed: int,
    pois_lambda: float = 1.0,
) -> tuple[MCMCResult, NDArray[np.bool_]]:
    """Simulate and fit one scenario under a deterministic profile."""
    rng_data, rng_fit = spawn_rngs(seed, 2)
    data = simulate_periodic_bernoulli(scenario, rng=rng_data)
    prior = ModelPrior(
        N=scenario.period,
        l=scenario.min_segment_length,
        gamma=1.0,
        pois_lambda=pois_lambda,
    )
    cfg = RJConfig(iters=profile.iters, burn=profile.burn, thin=profile.thin, seed=None)
    result = WithinPeriodCore(prior).fit(data, cfg=cfg, rng=rng_fit)
    return result, data


def posterior_tau_probabilities(samples: Iterable[Tau]) -> dict[Tau, float]:
    """Estimate posterior tau probabilities from retained samples."""
    sample_list = list(samples)
    if not sample_list:
        return {}
    counts = Counter(sample_list)
    total = len(sample_list)
    return {tau: count / total for tau, count in counts.items()}


def posterior_segment_count_probabilities(samples: Iterable[Tau]) -> dict[int, float]:
    """Estimate posterior segment-count probabilities from retained samples."""
    sample_list = list(samples)
    if not sample_list:
        return {}
    counts = Counter(1 if len(tau) == 0 else len(tau) for tau in sample_list)
    total = len(sample_list)
    return {segment_count: count / total for segment_count, count in sorted(counts.items())}


def highest_posterior_credible_set_size(samples: Iterable[Tau], mass: float = 0.95) -> int:
    """Return the number of tau states needed for the empirical HPCS."""
    probs = sorted(posterior_tau_probabilities(samples).values(), reverse=True)
    running = 0.0
    for idx, prob in enumerate(probs, start=1):
        running += prob
        if running >= mass:
            return idx
    return len(probs)


def marginal_changepoint_mass(result: MCMCResult, period: int) -> list[float]:
    """Return marginal posterior changepoint mass at each period bin."""
    if not result.samples_tau:
        return [0.0] * period
    return (result.changepoint_hist.astype(float) / len(result.samples_tau)).tolist()


def exact_tau_posterior(
    scenario: PeriodicBernoulliScenario,
    data: NDArray[np.bool_],
    *,
    pois_lambda: float = 1.0,
    max_states: int = 5000,
) -> dict[Tau, float] | None:
    """Evaluate the exact posterior over tau when the enumerated state space is small."""
    upper_bound = 1 + sum(
        math.comb(scenario.period, segment_count)
        for segment_count in range(2, scenario.period // scenario.min_segment_length + 1)
    )
    if upper_bound > max_states:
        return None
    states = enumerate_valid_taus(scenario.period, scenario.min_segment_length)
    if len(states) > max_states:
        return None
    model = WithinPeriodCore(
        ModelPrior(
            N=scenario.period,
            l=scenario.min_segment_length,
            gamma=1.0,
            pois_lambda=pois_lambda,
        )
    )
    model._prepare_counts(data)
    log_posts = {state: model._log_posterior_tau(state) for state in states}
    max_log = max(log_posts.values())
    normalizer = sum(math.exp(value - max_log) for value in log_posts.values())
    return {
        state: math.exp(log_posts[state] - max_log) / normalizer
        for state in states
    }


def summarize_fit(
    scenario: PeriodicBernoulliScenario,
    result: MCMCResult,
) -> FitSummary:
    """Build a typed summary from retained posterior samples."""
    tau_probs = posterior_tau_probabilities(result.samples_tau)
    return FitSummary(
        scenario=scenario.name,
        source_scope=scenario.source_scope,
        period=scenario.period,
        days=scenario.days,
        min_segment_length=scenario.min_segment_length,
        true_boundaries=scenario.boundaries,
        mode_tau=result.mode_tau,
        true_tau_probability=tau_probs.get(scenario.boundaries, 0.0),
        hpcs_size_95=highest_posterior_credible_set_size(result.samples_tau),
        acceptance_rate=result.acceptance_rate,
        sample_count=len(result.samples_tau),
    )


def run_reproduction(profile: ReplicationProfile) -> dict[str, object]:
    """Run deterministic paper-style and MySense-extension reproduction summaries."""
    scenario_results = []
    segment_tables = []
    prior_sensitivity = []
    exact_checks = []
    rngs = spawn_rngs(profile.seed, len(paper_replication_scenarios()) + 1)

    scenarios = paper_replication_scenarios()
    sensitivity_names = (
        {scenario.name for scenario in scenarios}
        if profile.name == "research"
        else {"paper_monte_carlo_n24", "one_activity_window"}
    )

    for scenario, rng in zip(scenarios, rngs[:-1], strict=True):
        seed = int(rng.integers(0, 2**32 - 1))
        result, data = fit_scenario(scenario, profile, seed=seed)
        summary = summarize_fit(scenario, result)
        scenario_results.append(_summary_payload(summary, result, scenario.period))
        segment_tables.append(
            {
                "scenario": scenario.name,
                "posterior_segment_counts": posterior_segment_count_probabilities(
                    result.samples_tau
                ),
            }
        )
        exact = exact_tau_posterior(
            scenario,
            data,
            max_states=profile.exact_state_limit,
        )
        exact_checks.append(_exact_payload(scenario, exact, result))
        if scenario.name in sensitivity_names:
            for lam in profile.prior_lambdas:
                sensitivity_result, _ = fit_scenario(
                    scenario,
                    profile,
                    seed=seed,
                    pois_lambda=lam,
                )
                prior_sensitivity.append(
                    {
                        "scenario": scenario.name,
                        "pois_lambda": lam,
                        "mode_tau": list(sensitivity_result.mode_tau),
                        "posterior_segment_counts": posterior_segment_count_probabilities(
                            sensitivity_result.samples_tau
                        ),
                    }
                )

    mysense = mysense_sensor_example(rng=rngs[-1], seed=None)
    mysense_scenario = PeriodicBernoulliScenario(
        name="mysense_any_activity",
        source_scope=mysense.source_scope,
        period=mysense.period,
        days=mysense.days,
        min_segment_length=4,
        boundaries=(),
        probabilities=(float(np.mean(mysense.any_activity)),),
        description=mysense.description,
    )
    prior = ModelPrior(N=mysense.period, l=4, gamma=1.0, pois_lambda=1.0)
    cfg = RJConfig(iters=profile.iters, burn=profile.burn, thin=profile.thin, seed=None)
    mysense_fit = WithinPeriodCore(prior).fit(mysense.any_activity, cfg=cfg, rng=rngs[-1])
    mysense_summary = summarize_fit(mysense_scenario, mysense_fit)

    return {
        "profile": asdict(profile),
        "profile_notes": [
            "The ci profile is a deterministic execution smoke test with short chains.",
            "Use the research profile for interpretable posterior summaries and prior sensitivity.",
        ],
        "paper_consistent": scenario_results,
        "mysense_extension": {
            "summary": _summary_payload(mysense_summary, mysense_fit, mysense.period),
            "sensor_rates": _sensor_rates(mysense),
        },
        "posterior_segment_tables": segment_tables,
        "prior_sensitivity": prior_sensitivity,
        "exact_checks": exact_checks,
        "discrepancies": [
            "The proprietary passive-sensor records from Taylor et al. are not bundled.",
            "Accessible article text gives core tables, but this script reproduces deterministic analogues rather than the original 1000-replication study.",
            "One-based paper event positions are converted to zero-based periodic bin-end indices.",
        ],
    }


def _summary_payload(
    summary: FitSummary,
    result: MCMCResult,
    period: int,
) -> dict[str, object]:
    """Convert a fit summary to a JSON-ready payload."""
    payload = asdict(summary)
    payload["true_boundaries"] = list(summary.true_boundaries)
    payload["mode_tau"] = list(summary.mode_tau)
    payload["marginal_changepoint_mass"] = marginal_changepoint_mass(result, period)
    payload["move_counts"] = result.move_counts
    return payload


def _exact_payload(
    scenario: PeriodicBernoulliScenario,
    exact: dict[Tau, float] | None,
    result: MCMCResult,
) -> dict[str, object]:
    """Summarize exact posterior comparison when available."""
    if exact is None:
        return {
            "scenario": scenario.name,
            "status": "skipped_state_space_too_large",
        }
    empirical = posterior_tau_probabilities(result.samples_tau)
    common_states = set(exact) | set(empirical)
    l1_error = sum(abs(exact.get(state, 0.0) - empirical.get(state, 0.0)) for state in common_states)
    exact_mode = max(exact.items(), key=lambda item: item[1])[0]
    return {
        "scenario": scenario.name,
        "status": "evaluated",
        "state_count": len(exact),
        "exact_mode": list(exact_mode),
        "exact_mode_probability": exact[exact_mode],
        "empirical_l1_error": l1_error,
    }


def _sensor_rates(series: SensorSeries) -> list[dict[str, float | str]]:
    """Compute average activity rates for each synthetic sensor."""
    return [
        {"sensor": name, "activity_rate": float(np.mean(values))}
        for name, values in sorted(series.sensors.items())
    ]


def write_reproduction_artifacts(
    output_dir: str | Path,
    *,
    profile: ReplicationProfile | str = CI_PROFILE,
) -> dict[str, Path]:
    """Run reproduction and write JSON, CSV, and SVG artifacts."""
    selected_profile = profile_by_name(profile) if isinstance(profile, str) else profile
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = run_reproduction(selected_profile)

    summary_path = out / "within_period_reproduction_summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    scenario_csv = out / "paper_scenario_summary.csv"
    _write_csv(
        scenario_csv,
        report["paper_consistent"],  # type: ignore[arg-type]
        [
            "scenario",
            "source_scope",
            "period",
            "days",
            "min_segment_length",
            "true_boundaries",
            "mode_tau",
            "true_tau_probability",
            "hpcs_size_95",
            "acceptance_rate",
            "sample_count",
        ],
    )

    sensitivity_csv = out / "prior_sensitivity.csv"
    _write_csv(
        sensitivity_csv,
        report["prior_sensitivity"],  # type: ignore[arg-type]
        ["scenario", "pois_lambda", "mode_tau", "posterior_segment_counts"],
    )

    sensor_csv = out / "mysense_sensor_rates.csv"
    mysense_payload = report["mysense_extension"]  # type: ignore[assignment]
    _write_csv(
        sensor_csv,
        mysense_payload["sensor_rates"],  # type: ignore[index,arg-type]
        ["sensor", "activity_rate"],
    )

    figure_path = out / "paper_changepoint_mass.svg"
    _write_mass_svg(figure_path, report["paper_consistent"])  # type: ignore[arg-type]

    return {
        "summary": summary_path,
        "paper_scenario_summary": scenario_csv,
        "prior_sensitivity": sensitivity_csv,
        "mysense_sensor_rates": sensor_csv,
        "paper_changepoint_mass": figure_path,
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: list[str]) -> None:
    """Write simple CSV rows with JSON encoding for structured fields."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row.get(field), sort_keys=True)
                    if isinstance(row.get(field), (dict, list, tuple))
                    else row.get(field)
                    for field in fields
                }
            )


def _write_mass_svg(path: Path, summaries: Iterable[Mapping[str, object]]) -> None:
    """Write a dependency-free SVG of marginal changepoint mass by scenario."""
    rows = list(summaries)
    width = 900
    row_height = 90
    height = 40 + row_height * len(rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 900 {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="24" font-family="Arial" font-size="16">Marginal changepoint mass</text>',
    ]
    for row_idx, row in enumerate(rows):
        y = 52 + row_idx * row_height
        period = int(row["period"])
        masses = list(row["marginal_changepoint_mass"])  # type: ignore[arg-type]
        label = str(row["scenario"])
        parts.append(f'<text x="20" y="{y}" font-family="Arial" font-size="12">{label}</text>')
        for idx, mass in enumerate(masses):
            x = 190 + idx * (680 / period)
            bar_height = max(1.0, float(mass) * 56.0)
            parts.append(
                f'<rect x="{x:.2f}" y="{y + 8 + 56 - bar_height:.2f}" '
                f'width="{max(1.0, 640 / period):.2f}" height="{bar_height:.2f}" '
                'fill="#2563eb"/>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
