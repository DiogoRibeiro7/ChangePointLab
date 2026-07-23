from __future__ import annotations

from importlib import import_module
from types import ModuleType


def import_optional(module: str, *, package: str, extra: str, feature: str) -> ModuleType:
    """Import an optional dependency with an actionable installation message."""
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name or module
        if missing == package or missing.startswith(f"{package}."):
            raise ImportError(
                f"{feature} requires optional dependency '{package}'. "
                f"Install it with `pip install 'changepoint-lab[{extra}]'` "
                f"or `poetry install --extras {extra}`."
            ) from exc
        raise


def require_matplotlib_pyplot(feature: str, *, backend: str | None = None) -> ModuleType:
    """Return ``matplotlib.pyplot`` for optional plotting features."""
    if backend is not None:
        matplotlib = import_optional(
            "matplotlib",
            package="matplotlib",
            extra="plot",
            feature=feature,
        )
        matplotlib.use(backend, force=True)
    return import_optional(
        "matplotlib.pyplot",
        package="matplotlib",
        extra="plot",
        feature=feature,
    )


def require_pandas(feature: str) -> ModuleType:
    """Return ``pandas`` for optional data-frame based I/O features."""
    return import_optional(
        "pandas",
        package="pandas",
        extra="data",
        feature=feature,
    )
