"""Utility helpers shared across statistical modules."""
from __future__ import annotations

from numpy.random import Generator, default_rng


def rng_from_state(random_state: Generator | int | None) -> Generator:
    """Create a :class:`numpy.random.Generator` from a seed or existing generator."""
    if isinstance(random_state, Generator):
        seed = int(random_state.integers(0, 2**63 - 1))
        return default_rng(seed)
    if random_state is None:
        return default_rng()
    return default_rng(random_state)
