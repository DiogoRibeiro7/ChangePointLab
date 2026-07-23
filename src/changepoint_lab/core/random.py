from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")

def make_rng(
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.random.Generator:
    """Return an owned NumPy generator from either a seed or an existing generator."""
    if seed is not None and rng is not None:
        raise ValueError("Pass either seed or rng, not both.")
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def spawn_rngs(seed: int | None, n_streams: int) -> tuple[np.random.Generator, ...]:
    """Create independent generator streams for parallel chains or resamplers."""
    if n_streams < 1:
        raise ValueError("n_streams must be positive.")
    seed_sequence = np.random.SeedSequence(seed)
    return tuple(np.random.default_rng(child) for child in seed_sequence.spawn(n_streams))


def choose_from_sequence(
    values: Sequence[T],
    rng: np.random.Generator,
) -> T:
    """Choose one item from a non-empty Python sequence using a NumPy generator."""
    if len(values) == 0:
        raise ValueError("cannot choose from an empty sequence.")
    return values[int(rng.integers(0, len(values)))]
