
"""Utilities for constructing walk-forward evaluation windows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for rolling walk-forward evaluation windows."""

    train_size: int
    val_size: int
    test_size: int
    step: int | None = None
    embargo: int = 0

    def validate(self) -> None:
        if self.train_size <= 0 or self.val_size <= 0 or self.test_size <= 0:
            raise ValueError("train_size, val_size and test_size must be positive integers")
        if self.embargo < 0:
            raise ValueError("embargo cannot be negative")
        if self.step is not None and self.step <= 0:
            raise ValueError("step must be a positive integer when provided")

    @property
    def window_length(self) -> int:
        """Total length of a single walk-forward window including embargo gaps."""

        return (
            self.train_size
            + self.embargo
            + self.val_size
            + self.embargo
            + self.test_size
        )

    def effective_step(self) -> int:
        """Return the stride between consecutive windows."""

        return self.step or self.test_size


@dataclass(frozen=True)
class WalkForwardWindow:
    """Indices representing a single walk-forward split."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def as_slices(self) -> tuple[slice, slice, slice]:
        """Return contiguous slices for each partition when possible."""

        def _to_slice(indices: np.ndarray) -> slice:
            if indices.size == 0:
                return slice(0, 0)
            return slice(int(indices[0]), int(indices[-1]) + 1)

        return _to_slice(self.train), _to_slice(self.val), _to_slice(self.test)

    def as_dict(self) -> dict[str, np.ndarray]:
        """Expose the split indices as a mapping for search APIs."""

        return {"train": self.train, "val": self.val, "test": self.test}

    def resolve(self, index: Sequence[pd.Timestamp]) -> dict[str, pd.DatetimeIndex]:
        """Map integer indices back to timestamps using the provided index."""

        ts_index = pd.DatetimeIndex(index)
        return {
            "train": ts_index[self.train],
            "val": ts_index[self.val],
            "test": ts_index[self.test],
        }


class WalkForwardScheduler:
    """Generate rolling walk-forward windows with embargo handling."""

    def __init__(self, config: WalkForwardConfig):
        self.cfg = config
        self.cfg.validate()

    def generate(self, index: Sequence[pd.Timestamp]) -> List[WalkForwardWindow]:
        """Generate windows covering the provided chronological index."""

        if isinstance(index, pd.DatetimeIndex):
            ordered_index = index.sort_values()
        else:
            ordered_index = pd.DatetimeIndex(index).sort_values()
        n = len(ordered_index)
        step = self.cfg.effective_step()
        windows: List[WalkForwardWindow] = []
        start = 0
        while True:
            train_start = start
            train_end = train_start + self.cfg.train_size
            val_start = train_end + self.cfg.embargo
            val_end = val_start + self.cfg.val_size
            test_start = val_end + self.cfg.embargo
            test_end = test_start + self.cfg.test_size
            if test_end > n:
                break
            train_idx = np.arange(train_start, train_end, dtype=int)
            val_idx = np.arange(val_start, val_end, dtype=int)
            test_idx = np.arange(test_start, test_end, dtype=int)
            windows.append(WalkForwardWindow(train=train_idx, val=val_idx, test=test_idx))
            start += step
            if start + self.cfg.window_length > n:
                break
        if not windows:
            raise ValueError(
                "Unable to construct walk-forward windows with the provided configuration and index"
            )
        return windows

    def splits_for(self, index: Sequence[pd.Timestamp]) -> List[dict[str, np.ndarray]]:
        """Return windows as dictionaries suitable for hyperparameter search APIs."""

        return [window.as_dict() for window in self.generate(index)]

    def __call__(self, index: Sequence[pd.Timestamp]) -> List[WalkForwardWindow]:
        return self.generate(index)
