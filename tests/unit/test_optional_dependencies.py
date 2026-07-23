from __future__ import annotations

import subprocess
import sys


def _run_blocked_imports(code: str) -> subprocess.CompletedProcess[str]:
    wrapper = f"""
import importlib.abc
import sys


class Blocker(importlib.abc.MetaPathFinder):
    blocked = ("matplotlib", "pandas")

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.blocked or fullname.startswith(tuple(name + "." for name in self.blocked)):
            raise ModuleNotFoundError(f"No module named '{{fullname}}'", name=fullname)
        return None


sys.meta_path.insert(0, Blocker())
{code}
"""
    return subprocess.run(
        [sys.executable, "-c", wrapper],
        capture_output=True,
        text=True,
    )


def test_core_imports_without_plot_or_data_extras() -> None:
    result = _run_blocked_imports(
        """
import numpy as np
import changepoint_lab as cpl
from changepoint_lab.algorithms.optimization.pelt import NormalMeanVarUnknown, bic_penalty

x = np.r_[np.zeros(6), np.ones(6)]
cost = NormalMeanVarUnknown()
cost.precompute(x)
out = cpl.PELT(cost_fn=cost, penalty=bic_penalty(2, len(x))).fit_predict(x)
assert out.indices.ndim == 1
assert cpl.edivisive(x.reshape(-1, 1), min_size=2, R=3, resample="iid").change_points is not None
"""
    )
    assert result.returncode == 0, result.stderr


def test_plotting_extra_error_is_actionable_when_matplotlib_missing() -> None:
    result = _run_blocked_imports(
        """
import numpy as np
from changepoint_lab.algorithms.bayesian.bocpd.plotting import plot_cp_probability

try:
    plot_cp_probability(np.array([0.1, 0.2]))
except ImportError as exc:
    message = str(exc)
    assert "changepoint-lab[plot]" in message
    assert "poetry install --extras plot" in message
else:
    raise AssertionError("plotting unexpectedly succeeded")
"""
    )
    assert result.returncode == 0, result.stderr


def test_data_extra_error_is_actionable_when_pandas_missing(tmp_path) -> None:
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("timestamp\n2026-01-01T00:00:00\n")
    result = _run_blocked_imports(
        f"""
from changepoint_lab.common.io.data_loader import load_binary_from_csv

try:
    load_binary_from_csv(r"{csv_path}")
except ImportError as exc:
    message = str(exc)
    assert "changepoint-lab[data]" in message
    assert "poetry install --extras data" in message
else:
    raise AssertionError("CSV loading unexpectedly succeeded")
"""
    )
    assert result.returncode == 0, result.stderr


def test_cli_help_without_optional_extras() -> None:
    result = _run_blocked_imports(
        """
from changepoint_lab.cli.cpd_cli import create_parser

parser = create_parser()
help_text = parser.format_help()
assert "ChangePointLab CLI" in help_text
"""
    )
    assert result.returncode == 0, result.stderr
