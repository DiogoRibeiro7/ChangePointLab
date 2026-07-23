from __future__ import annotations

import argparse
import json
from pathlib import Path

from changepoint_lab.algorithms.bayesian.within_period import write_reproduction_artifacts


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run within-period Taylor-style and MySense-extension reproduction artifacts."
    )
    parser.add_argument(
        "--profile",
        choices=("ci", "research"),
        default="ci",
        help="Execution profile. 'ci' is deterministic and short; 'research' uses longer chains.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "within_period_reproduction",
        help="Directory for JSON, CSV, and SVG outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the reproduction profile and print generated artifact paths."""
    args = parse_args()
    artifacts = write_reproduction_artifacts(args.output, profile=args.profile)
    print(json.dumps({key: str(path) for key, path in artifacts.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
