#!/usr/bin/env python3
"""Capture command-level baseline evidence for the current checkout."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "audit"
REPORT_PATH = ROOT / "docs" / "audit" / "baseline_test_truth.md"


def _run(args: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    merged_env = os.environ.copy()
    merged_env.setdefault("MPLBACKEND", "Agg")
    if env:
        merged_env.update(env)
    proc = subprocess.run(args, cwd=cwd, env=merged_env, capture_output=True, text=True)
    duration = time.perf_counter() - started
    output = f"{proc.stdout}\n{proc.stderr}".strip()
    return {
        "command": args,
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "duration_seconds": round(duration, 3),
        "summary": _summarize_output(output),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _summarize_output(output: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    patterns = {
        "failed": r"(\d+)\s+failed",
        "passed": r"(\d+)\s+passed",
        "skipped": r"(\d+)\s+skipped",
        "warnings": r"(\d+)\s+warnings?",
        "errors": r"(\d+)\s+errors?",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, output, flags=re.IGNORECASE)
        if matches:
            summary[key] = int(matches[-1])
    if "no tests ran" in output.lower():
        summary["no_tests_ran"] = True
    if "error:" in output.lower() and "errors" not in summary:
        summary["contains_error_text"] = True
    return summary


def _git_value(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _python_probe() -> dict[str, Any]:
    code = """
import importlib.metadata as md
import json
import numpy as np
import platform
import sys

packages = ["changepoint-lab", "numpy", "pytest", "ruff", "mypy", "pydocstyle", "sphinx", "pdoc"]
versions = {}
for package in packages:
    try:
        versions[package] = md.version(package)
    except md.PackageNotFoundError:
        versions[package] = None

print(json.dumps({
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "numpy": np.__version__,
    "packages": versions,
}, sort_keys=True))
"""
    proc = _run([sys.executable, "-c", code])
    if proc["returncode"] != 0:
        return {"probe_error": proc}
    return json.loads(proc["stdout"])


def _write_fixture_files(tmp: Path) -> dict[str, Path]:
    counts = tmp / "counts.csv"
    counts.write_text("count\n0\n0\n1\n1\n0\n0\n", encoding="utf-8")
    return {"counts": counts}


def _stable_export_code() -> str:
    return """
import changepoint_lab as cpl

missing = []
for name in cpl.__all__:
    try:
        getattr(cpl, name)
    except Exception as exc:
        missing.append((name, type(exc).__name__, str(exc)))
if missing:
    raise SystemExit(repr(missing))
print(f"stable exports ok: {len(cpl.__all__)}")
"""


def _tiny_execution_code() -> str:
    return """
import numpy as np
from changepoint_lab import BOCPD, EDivisive, KernelCPD, PELT
from changepoint_lab.algorithms.bayesian.bocpd import BOCPDConfig, ConstantHazard
from changepoint_lab.algorithms.optimization.pelt import NormalMeanKnownVar

x = np.r_[np.zeros(4), np.ones(4)]
cost = NormalMeanKnownVar(sigma2=1.0)
result = PELT(cost_fn=cost, penalty=1.0, min_seg_len=2).fit_predict(x)
assert result.indices.ndim == 1
assert KernelCPD(penalty=0.1).fit_predict(x.reshape(-1, 1)).indices.ndim == 1
assert EDivisive(min_size=2, R=3, seed=0).fit_predict(x.reshape(-1, 1)).indices.ndim == 1
stream = np.array([0, 0, 1, 1], dtype=bool)
bocpd = BOCPD(ConstantHazard(mean_run_length=4), BOCPDConfig(max_run_length=8))
assert bocpd.run(stream).cp_prob.shape == stream.shape
print("tiny executions ok")
"""


def _installed_wheel_code(tmp: Path) -> str:
    smoke_code = _stable_export_code() + "\n" + _tiny_execution_code()
    return """
import pathlib
import subprocess
import sys
import venv

root = pathlib.Path({root!r})
tmp = pathlib.Path({tmp!r})
wheel = sorted((root / "dist").glob("*.whl"))[-1]
venv_dir = tmp / "wheel-venv"
venv.EnvBuilder(with_pip=True, system_site_packages=True).create(venv_dir)
python = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
subprocess.check_call([str(python), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)])
code = {code!r}
subprocess.check_call([str(python), "-c", code], cwd=tmp)
print(f"installed wheel smoke ok: {{wheel.name}}")
""".format(root=str(ROOT), tmp=str(tmp), code=smoke_code)


def _command_plan(tmp: Path) -> list[tuple[str, list[str]]]:
    files = _write_fixture_files(tmp)
    py = sys.executable
    commands: list[tuple[str, list[str]]] = [
        ("unit_tests", [py, "-m", "pytest", "-m", "not slow", "tests/unit"]),
        ("integration_tests", [py, "-m", "pytest", "-m", "integration", "tests/integration"]),
        ("slow_tests", [py, "-m", "pytest", "-m", "slow", "tests"]),
        ("benchmark_tests", [py, "-m", "pytest", "tests/unit/test_performance.py"]),
        ("ruff", [py, "-m", "ruff", "check", "."]),
        ("mypy", [py, "-m", "mypy"]),
        ("pydocstyle", [py, "-m", "pydocstyle", "src/changepoint_lab"]),
        ("sphinx_docs", ["sphinx-build", "-b", "html", "docs", "docs/_build/html"]),
        ("pdoc_docs", [py, "-m", "pdoc", "-o", "docs/_build/pdoc", "changepoint_lab"]),
        ("package_build", ["poetry", "build"]),
        ("distribution_validation", [py, "scripts/validate_distribution.py", "dist"]),
        ("installed_wheel_smoke", [py, "-c", _installed_wheel_code(tmp)]),
        ("fresh_process_import", [py, "-c", "import changepoint_lab; print(changepoint_lab.__version__)"]),
        ("stable_top_level_exports", [py, "-c", _stable_export_code()]),
        ("tiny_public_executions", [py, "-c", _tiny_execution_code()]),
        ("cpd_help", [py, "-m", "changepoint_lab.cli.cpd_cli", "--help"]),
        ("bocpd_help", [py, "-m", "changepoint_lab.cli.bocpd_cli", "--help"]),
        ("within_period_help", [py, "-m", "changepoint_lab.algorithms.bayesian.within_period.cli", "--help"]),
        (
            "cpd_tiny_execution",
            [
                py,
                "-m",
                "changepoint_lab.cli.cpd_cli",
                "--input",
                str(files["counts"]),
                "--output",
                str(tmp / "cpd-out"),
                "edivisive",
                "--columns",
                "count",
                "--min-size",
                "2",
                "--R",
                "3",
            ],
        ),
        (
            "bocpd_tiny_execution",
            [
                py,
                "-m",
                "changepoint_lab.cli.bocpd_cli",
                "--demo",
                "--days",
                "2",
                "--period",
                "4",
                "--Rmax",
                "8",
                "--outdir",
                str(tmp / "bocpd-out"),
            ],
        ),
        (
            "within_period_tiny_execution",
            [
                py,
                "-m",
                "changepoint_lab.algorithms.bayesian.within_period.cli",
                "--demo",
                "--N",
                "8",
                "--l",
                "2",
                "--days",
                "2",
                "--iters",
                "20",
                "--burn",
                "5",
                "--thin",
                "5",
                "--outdir",
                str(tmp / "within-out"),
            ],
        ),
    ]
    if shutil.which("poetry") is None:
        commands = [(name, cmd) for name, cmd in commands if cmd[0] != "poetry"]
    if shutil.which("sphinx-build") is None:
        commands = [(name, cmd) for name, cmd in commands if name != "sphinx_docs"]
    return commands


def _write_report(data: dict[str, Any], report_path: Path) -> None:
    results = data["commands"]
    lines = [
        "# Baseline test truth",
        "",
        f"Captured on: `{data['captured_at']}`",
        f"Git commit: `{data['git']['commit']}`",
        f"Branch: `{data['git']['branch']}`",
        f"Python: `{data['environment'].get('python', 'unknown').splitlines()[0]}`",
        f"NumPy: `{data['environment'].get('numpy', 'unknown')}`",
        f"Platform: `{data['environment'].get('platform', 'unknown')}`",
        "",
        "This audit records the current executable state before scientific behavior changes.",
        "Passing checks do not upgrade any scientific verification status.",
        "",
        "## Command Results",
        "",
        "| Check | Result | Duration | Parsed Summary |",
        "| --- | --- | ---: | --- |",
    ]
    for name, result in results.items():
        status = "pass" if result["returncode"] == 0 else f"fail ({result['returncode']})"
        summary = json.dumps(result["summary"], sort_keys=True)
        lines.append(
            f"| `{name}` | {status} | {result['duration_seconds']:.3f}s | `{summary}` |"
        )
    failures = [name for name, result in results.items() if result["returncode"] != 0]
    lines.extend(["", "## Failure Characterization", ""])
    if failures:
        for name in failures:
            result = results[name]
            tail = "\n".join((result["stdout"] + result["stderr"]).splitlines()[-20:])
            lines.extend([f"### `{name}`", "", "```text", tail, "```", ""])
    else:
        lines.append("No command failures were observed in this capture.")
    lines.extend(
        [
            "",
            "## Remaining Scientific Limitations",
            "",
            "- This change records executable evidence only.",
            "- Existing partial or unverified methods remain at their current verification status.",
            "- Follow-up changes should address correctness, API, documentation, and release-readiness items separately.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def capture() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cpl-audit-") as tmp_name:
        tmp = Path(tmp_name)
        commands = {}
        for name, cmd in _command_plan(tmp):
            commands[name] = _run(cmd)
    return {
        "schema_version": "1.0",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": _git_value(["rev-parse", "HEAD"]),
            "branch": _git_value(["branch", "--show-current"]),
            "status_short": _git_value(["status", "--short"]),
        },
        "environment": _python_probe(),
        "commands": commands,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_DIR / "baseline_test_truth.json",
        help="Machine-readable output path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_PATH,
        help="Human-readable report path.",
    )
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    data = capture()
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(data, args.report)

    failed = [name for name, result in data["commands"].items() if result["returncode"] != 0]
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")
    if failed:
        print("recorded failures: " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
