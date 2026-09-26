"""Translate filesystem failures at artifact write boundaries."""

from __future__ import annotations

import csv
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from dataexcept import DataLoadingError, FileWriteError


@contextmanager
def reading(path: str | Path) -> Iterator[None]:
    """Preserve the failed input path and original read or decode error."""
    try:
        yield
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DataLoadingError(str(path), exc) from exc


@contextmanager
def writing(path: str | Path) -> Iterator[None]:
    """Preserve the failed output path and original filesystem error."""
    try:
        yield
    except (OSError, UnicodeError) as exc:
        raise FileWriteError(str(path), exc) from exc
