"""CSV input failures keep the source and preserve their original causes."""

from __future__ import annotations

import pandas as pd
import pytest
from dataexcept import DataLoadingError, SchemaMismatchError

from changepoint_lab.common.io.data_loader import load_binary_from_csv


def test_missing_csv_preserves_os_error(tmp_path) -> None:
    missing = tmp_path / "missing.csv"

    with pytest.raises(DataLoadingError) as caught:
        load_binary_from_csv(missing)

    assert caught.value.source == str(missing)
    assert isinstance(caught.value.original, FileNotFoundError)
    assert caught.value.__cause__ is caught.value.original


def test_invalid_encoding_preserves_decode_error(tmp_path) -> None:
    path = tmp_path / "invalid.csv"
    path.write_bytes(b"timestamp\n\xff\n")

    with pytest.raises(DataLoadingError) as caught:
        load_binary_from_csv(path)

    assert caught.value.source == str(path)
    assert isinstance(caught.value.original, UnicodeDecodeError)


def test_malformed_csv_preserves_parser_error(tmp_path) -> None:
    path = tmp_path / "malformed.csv"
    path.write_text('timestamp,value\n"unclosed,1\n')

    with pytest.raises(DataLoadingError) as caught:
        load_binary_from_csv(path)

    assert caught.value.source == str(path)
    assert isinstance(caught.value.original, pd.errors.ParserError)


@pytest.mark.parametrize("column,kwargs", [
    ("timestamp", {}),
    ("value", {"timestamp_col": "seen", "value_col": "value"}),
])
def test_missing_columns_are_schema_errors(tmp_path, column, kwargs) -> None:
    path = tmp_path / "events.csv"
    path.write_text("seen\n2026-01-01\n")

    with pytest.raises(SchemaMismatchError) as caught:
        load_binary_from_csv(path, **kwargs)

    assert column in caught.value.expected
    assert "seen" in caught.value.found


def test_invalid_options_remain_value_errors(tmp_path) -> None:
    with pytest.raises(ValueError, match="bin_minutes"):
        load_binary_from_csv(tmp_path / "missing.csv", bin_minutes=7)
