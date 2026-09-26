"""File boundaries retain paths and causes without hiding input validation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from dataexcept import DataLoadingError, FileWriteError, SchemaMismatchError

from changepoint_lab.algorithms.bayesian.within_period.replication import (
    write_reproduction_artifacts,
)
from changepoint_lab.cli.cpd_cli import load_csv_data, save_results
from changepoint_lab.common.io.io_utils import load_result_npz, save_result_npz


def _save_run(path) -> None:
    prior = SimpleNamespace(N=12, l=2, gamma=0.1, pois_lambda=1.0)
    config = SimpleNamespace(
        iters=10, burn=2, thin=2, seed=None,
        move_prob=0.5, birth_prob=0.25, death_prob=0.25,
    )
    save_result_npz(
        path, samples_tau=[(3, 9), ()], log_posteriors=[-1.0, -2.0],
        changepoint_hist=np.zeros(12, dtype=int), mode_tau=(3, 9),
        prior_obj=prior, cfg_obj=config,
    )


def test_npz_round_trip(tmp_path) -> None:
    path = tmp_path / "run.npz"
    _save_run(path)

    loaded = load_result_npz(path)
    assert loaded["samples_tau"] == [(3, 9), ()]
    assert loaded["mode_tau"] == (3, 9)
    assert loaded["prior"]["N"] == 12
    assert loaded["cfg"]["seed"] is None


def test_npz_write_error_keeps_path_and_cause(tmp_path) -> None:
    path = tmp_path / "absent" / "run.npz"
    with pytest.raises(FileWriteError) as caught:
        _save_run(path)
    assert caught.value.path == str(path)
    assert isinstance(caught.value.original, FileNotFoundError)
    assert caught.value.__cause__ is caught.value.original


@pytest.mark.parametrize("contents", [None, b"invalid archive"])
def test_npz_read_error_keeps_path_and_cause(tmp_path, contents) -> None:
    path = tmp_path / "run.npz"
    if contents is not None:
        path.write_bytes(contents)
    with pytest.raises(DataLoadingError) as caught:
        load_result_npz(path)
    assert caught.value.source == str(path)
    assert caught.value.__cause__ is caught.value.original


def test_npz_missing_fields_is_schema_error(tmp_path) -> None:
    path = tmp_path / "incomplete.npz"
    np.savez_compressed(path, samples_flat=np.array([1]))
    with pytest.raises(SchemaMismatchError) as caught:
        load_result_npz(path)
    assert "samples_idx" in caught.value.expected
    assert "samples_flat" in caught.value.found


def test_cli_csv_input_errors(tmp_path) -> None:
    missing = tmp_path / "missing.csv"
    with pytest.raises(DataLoadingError) as caught:
        load_csv_data(str(missing))
    assert caught.value.source == str(missing)
    assert isinstance(caught.value.original, FileNotFoundError)

    empty = tmp_path / "empty.csv"
    empty.touch()
    with pytest.raises(SchemaMismatchError, match="CSV header"):
        load_csv_data(str(empty))

    valid = tmp_path / "valid.csv"
    valid.write_text("value\n1\n")
    with pytest.raises(SchemaMismatchError, match="other"):
        load_csv_data(str(valid), columns="other")


def test_cli_export_error_points_to_file(tmp_path) -> None:
    class FailedFigure:
        def savefig(self, path, **kwargs):
            raise PermissionError("unwritable image")

    with pytest.raises(FileWriteError) as caught:
        save_results(tmp_path, {}, {"plot": FailedFigure()}, "demo")
    assert caught.value.path == str(tmp_path / "demo_plot.png")
    assert isinstance(caught.value.original, PermissionError)


def test_replication_directory_error_is_typed(tmp_path) -> None:
    occupied = tmp_path / "occupied"
    occupied.write_text("file instead of directory")
    with pytest.raises(FileWriteError) as caught:
        write_reproduction_artifacts(occupied)
    assert caught.value.path == str(occupied)
    assert isinstance(caught.value.original, FileExistsError)
