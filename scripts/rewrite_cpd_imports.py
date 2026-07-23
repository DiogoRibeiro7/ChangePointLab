#!/usr/bin/env python3
"""Rewrite legacy ``cpd.*`` imports to new package locations.

This script walks through Python files and updates import statements based on a
hardcoded mapping from old module paths to new ones. It uses ``libcst`` to
safely rewrite the syntax tree without touching unrelated imports.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Iterable

import libcst as cst
from libcst.helpers import get_full_name_for_node

# Deterministic mapping from old to new module paths
MODULE_MAP = {
    "cpd.anchor_utils": "changepoint_lab.algorithms.bayesian.within_period.anchor_utils",
    "cpd.within_period_cpd": "changepoint_lab.algorithms.bayesian.within_period.within_period_cpd",
    "cpd.posterior_predictive": "changepoint_lab.algorithms.bayesian.within_period.posterior_predictive",
    "cpd.cli": "changepoint_lab.algorithms.bayesian.within_period.cli",
    "cpd.tempering": "changepoint_lab.algorithms.bayesian.within_period.samplers.tempering",
    "cpd.rff_variants": "changepoint_lab.algorithms.kernel.rff_variants",
    "cpd.bandwidth_cv": "changepoint_lab.algorithms.kernel.bandwidth_cv",
    "cpd.ar_emissions": "changepoint_lab.algorithms.state_space.emissions.ar_emissions",
    "cpd.gaussian_full": "changepoint_lab.algorithms.state_space.emissions.gaussian_full",
    "cpd.cpd_cli": "changepoint_lab.cli.cpd_cli",
    "cpd.api_harmonizer": "changepoint_lab.common.api_harmonizer",
    "cpd.data_loader": "changepoint_lab.common.io.data_loader",
    "cpd.io_utils": "changepoint_lab.common.io.io_utils",
    "cpd.plotting_helpers": "changepoint_lab.common.plotting.plotting_helpers",
    "cpd.diagnostics": "changepoint_lab.common.diagnostics.diagnostics",
    "cpd.types": "changepoint_lab.common.types.types",
    "cpd.utils": "changepoint_lab.common.utils.utils",
    # Bare names that previously relied on being in the same folder
    "anchor_utils": "changepoint_lab.algorithms.bayesian.within_period.anchor_utils",
    "within_period_cpd": "changepoint_lab.algorithms.bayesian.within_period.within_period_cpd",
    "posterior_predictive": "changepoint_lab.algorithms.bayesian.within_period.posterior_predictive",
    "tempering": "changepoint_lab.algorithms.bayesian.within_period.samplers.tempering",
    "rff_variants": "changepoint_lab.algorithms.kernel.rff_variants",
    "bandwidth_cv": "changepoint_lab.algorithms.kernel.bandwidth_cv",
    "ar_emissions": "changepoint_lab.algorithms.state_space.emissions.ar_emissions",
    "gaussian_full": "changepoint_lab.algorithms.state_space.emissions.gaussian_full",
    "cpd_cli": "changepoint_lab.cli.cpd_cli",
    "api_harmonizer": "changepoint_lab.common.api_harmonizer",
    "data_loader": "changepoint_lab.common.io.data_loader",
    "io_utils": "changepoint_lab.common.io.io_utils",
    "plotting_helpers": "changepoint_lab.common.plotting.plotting_helpers",
    "diagnostics": "changepoint_lab.common.diagnostics.diagnostics",
    "types": "changepoint_lab.common.types.types",
    "utils": "changepoint_lab.common.utils.utils",
}


class ImportRewriter(cst.CSTTransformer):
    """CST transformer that rewrites imports using ``MODULE_MAP``."""

    def leave_Import(self, original_node: cst.Import, updated_node: cst.Import) -> cst.Import:
        new_names = []
        changed = False
        for alias in updated_node.names:
            full_name = get_full_name_for_node(alias.name)
            if full_name in MODULE_MAP:
                new_names.append(alias.with_changes(name=cst.parse_expression(MODULE_MAP[full_name])))
                changed = True
            else:
                new_names.append(alias)
        return updated_node.with_changes(names=new_names) if changed else updated_node

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom:
        # Skip relative imports (e.g. ``from .foo import bar``)
        if updated_node.relative:
            return updated_node
        module = updated_node.module
        if module is None:
            return updated_node
        full_name = get_full_name_for_node(module)
        if full_name in MODULE_MAP:
            new_module = cst.parse_expression(MODULE_MAP[full_name])
            return updated_node.with_changes(module=new_module)
        return updated_node


def iter_py_files(paths: Iterable[pathlib.Path]) -> Iterable[pathlib.Path]:
    for path in paths:
        if path.is_dir():
            yield from (p for p in path.rglob("*.py") if p.is_file())
        elif path.suffix == ".py":
            yield path


def rewrite_file(path: pathlib.Path) -> bool:
    source = path.read_text()
    module = cst.parse_module(source)
    new_module = module.visit(ImportRewriter())
    if new_module.code != source:
        path.write_text(new_module.code)
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=pathlib.Path, default=[pathlib.Path(".")])
    args = parser.parse_args()
    changed_any = False
    for py_file in iter_py_files(args.paths):
        if rewrite_file(py_file):
            changed_any = True
    return 0 if changed_any else 0


if __name__ == "__main__":
    sys.exit(main())
