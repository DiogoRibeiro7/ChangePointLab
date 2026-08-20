#!/usr/bin/env python3
"""Smoke-test a built wheel with only core runtime dependencies installed."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _find_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one wheel in {dist_dir}, found {len(wheels)}")
    return wheels[0]


def _venv_python(env_dir: Path) -> Path:
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _run(command: list[str], cwd: Path) -> None:
    subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def smoke_wheel(wheel: Path, numpy_spec: str) -> None:
    with tempfile.TemporaryDirectory(prefix="cpl-core-wheel-") as tmp:
        tmp_path = Path(tmp)
        env_dir = tmp_path / "venv"
        venv.EnvBuilder(with_pip=True).create(env_dir)
        python = _venv_python(env_dir)

        _run([str(python), "-m", "pip", "install", "--upgrade", "pip"], tmp_path)
        _run([str(python), "-m", "pip", "install", numpy_spec], tmp_path)
        _run([str(python), "-m", "pip", "install", "--no-deps", str(wheel)], tmp_path)
        _run(
            [
                str(python),
                "-c",
                (
                    "import importlib.util; "
                    "import numpy as np; "
                    "import changepoint_lab as cpl; "
                    "from changepoint_lab.algorithms.optimization.pelt import "
                    "NormalMeanVarUnknown, bic_penalty; "
                    "assert importlib.util.find_spec('matplotlib') is None; "
                    "assert importlib.util.find_spec('pandas') is None; "
                    "x = np.r_[np.zeros(6), np.ones(6)]; "
                    "cost = NormalMeanVarUnknown(); "
                    "cost.precompute(x); "
                    "result = cpl.PELT(cost_fn=cost, penalty=bic_penalty(2, len(x))).fit_predict(x); "
                    "assert result.indices.ndim == 1; "
                    "assert cpl.__version__"
                ),
            ],
            tmp_path,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--numpy-spec", default="numpy")
    args = parser.parse_args(argv)

    wheel = args.wheel if args.wheel is not None else _find_wheel(args.dist_dir)
    smoke_wheel(wheel.resolve(), args.numpy_spec)
    print("core wheel smoke validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
