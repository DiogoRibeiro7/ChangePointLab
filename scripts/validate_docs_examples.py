#!/usr/bin/env python3
"""Execute selected documentation examples against a built wheel."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MARKER = "<!-- docs-example: execute -->"


def _find_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one wheel in {dist_dir}, found {len(wheels)}")
    return wheels[0]


def _venv_python(env_dir: Path) -> Path:
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _marked_python_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(EXAMPLE_MARKER) + r"\s*```python\r?\n(?P<code>.*?)\r?\n```",
        re.DOTALL,
    )
    return [match.group("code") for match in pattern.finditer(text)]


def _install_wheel(python: Path, wheel: Path) -> None:
    subprocess.run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    subprocess.run(
        [str(python), "-m", "pip", "install", "numpy"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _run_code_block(python: Path, work_dir: Path, label: str, code: str) -> None:
    script = work_dir / f"{label}.py"
    script.write_text(code, encoding="utf-8")
    subprocess.run(
        [str(python), str(script)],
        cwd=work_dir,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def validate_examples(wheel: Path, docs: list[Path]) -> None:
    examples: list[tuple[str, str]] = []
    for doc in docs:
        blocks = _marked_python_blocks(doc)
        examples.extend((f"{doc.stem}_{index}", code) for index, code in enumerate(blocks, 1))

    if not examples:
        raise RuntimeError("no executable documentation examples found")

    with tempfile.TemporaryDirectory(prefix="cpl-docs-") as tmp:
        tmp_path = Path(tmp)
        env_dir = tmp_path / "venv"
        venv.EnvBuilder(with_pip=True).create(env_dir)
        python = _venv_python(env_dir)
        _install_wheel(python, wheel)

        subprocess.run(
            [
                str(python),
                "-c",
                "import changepoint_lab as cpl; assert cpl.__version__",
            ],
            cwd=tmp_path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for index, (label, code) in enumerate(examples, 1):
            _run_code_block(python, tmp_path, f"{index:02d}_{label}", code)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--wheel", type=Path)
    parser.add_argument(
        "--doc",
        type=Path,
        action="append",
        default=[ROOT / "README.md", ROOT / "docs" / "tutorials" / "getting_started_tutorial.md"],
        help="Markdown file containing marked executable Python blocks.",
    )
    args = parser.parse_args(argv)

    wheel = args.wheel if args.wheel is not None else _find_wheel(args.dist_dir)
    validate_examples(wheel.resolve(), [path.resolve() for path in args.doc])
    print("documentation examples validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
