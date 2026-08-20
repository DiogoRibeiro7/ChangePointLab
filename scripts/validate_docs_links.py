#!/usr/bin/env python3
"""Validate local Markdown documentation links."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\((?P<target>[^)]+)\)")
SKIP_SCHEMES = {"http", "https", "mailto"}


def _markdown_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".md":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.md")))
    return files


def _target_path(source: Path, target: str) -> Path | None:
    parsed = urlparse(target.strip())
    if parsed.scheme in SKIP_SCHEMES or target.startswith("#"):
        return None
    if parsed.scheme or parsed.netloc:
        return None

    raw_path = unquote(parsed.path)
    if not raw_path:
        return None
    if raw_path.startswith("/"):
        return ROOT / raw_path.lstrip("/")
    return source.parent / raw_path


def validate_links(paths: list[Path]) -> None:
    failures: list[str] = []
    for markdown_file in _markdown_files(paths):
        text = markdown_file.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK_RE.finditer(text):
            target = match.group("target").split()[0]
            local_path = _target_path(markdown_file, target)
            if local_path is not None and not local_path.exists():
                relative_source = markdown_file.relative_to(ROOT)
                relative_target = local_path.relative_to(ROOT)
                failures.append(f"{relative_source}: missing {relative_target}")

    if failures:
        details = "\n".join(failures)
        raise RuntimeError(f"broken documentation links:\n{details}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[ROOT / "README.md", ROOT / "docs"],
        help="Markdown file or directory paths to validate.",
    )
    args = parser.parse_args(argv)

    validate_links([path.resolve() for path in args.paths])
    print("documentation links validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
