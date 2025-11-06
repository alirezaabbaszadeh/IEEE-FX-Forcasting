"""Purged cross-validation utilities with embargo handling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedCVConfig:
    """Configuration for purged cross-validation splits."""

    n_splits: int
    test_size: int | None = None
    embargo: int = 0

    def validate(self, *, total_length: int | None = None) -> None:
        if self.n_splits <= 1:
            raise ValueError("n_splits must be greater than 1")
        if self.test_size is not None and self.test_size <= 0:
            raise ValueError("test_size must be a positive integer when provided")
        if self.embargo < 0:
            raise ValueError("embargo cannot be negative")
        if total_length is not None and self.test_size is not None:
            if self.test_size * self.n_splits > total_length:
                raise ValueError("Requested folds exceed available observations")


@dataclass(frozen=True)
class PurgedCVDiagnostics:
    """Describe gaps introduced around the validation fold."""

    test_range: tuple[pd.Timestamp, pd.Timestamp] | None
    left_gap: pd.Timedelta | None
    right_gap: pd.Timedelta | None


@dataclass(frozen=True)
class PurgedCVSplit:
    """Indices representing a purged cross-validation split."""

    fold: int
    train: np.ndarray
    test: np.ndarray
    diagnostics: PurgedCVDiagnostics

    def record(self) -> dict[str, object]:
        """Serialise the split for audit trails."""

        test_range = self.diagnostics.test_range
        return {
            "fold": self.fold,
            "split": "test",
            "size": int(self.test.size),
            "start": test_range[0].isoformat() if test_range else None,
            "end": test_range[1].isoformat() if test_range else None,
            "left_gap": str(self.diagnostics.left_gap) if self.diagnostics.left_gap else None,
            "right_gap": str(self.diagnostics.right_gap) if self.diagnostics.right_gap else None,
        }


class PurgedCVSplitter:
    """Generate purged cross-validation splits with an embargo around test folds."""

    def __init__(self, config: PurgedCVConfig):
        self.cfg = config

    def split(self, index: Sequence[pd.Timestamp]) -> List[PurgedCVSplit]:
        timestamps = pd.DatetimeIndex(index).sort_values()
        n = len(timestamps)
        if n < 2:
            raise ValueError("At least two observations are required for purged CV")
        self.cfg.validate(total_length=n)

        test_size = self.cfg.test_size or n // self.cfg.n_splits
        if test_size <= 0:
            raise ValueError("test_size resolves to zero; increase data length or specify a value")

        splits: List[PurgedCVSplit] = []
        for fold in range(self.cfg.n_splits):
            start = fold * test_size
            end = start + test_size
            if fold == self.cfg.n_splits - 1:
                end = n
            if start >= n:
                break
            test_idx = np.arange(start, min(end, n), dtype=int)
            embargo = self.cfg.embargo
            left_end = max(start - embargo, 0)
            right_start = min(end + embargo, n)
            left_train = np.arange(0, left_end, dtype=int)
            right_train = np.arange(right_start, n, dtype=int)
            train_idx = np.concatenate([left_train, right_train])

            test_range = None
            if test_idx.size:
                test_range = (timestamps[test_idx[0]], timestamps[test_idx[-1]])
            left_gap = None
            if left_train.size and test_idx.size:
                left_gap = timestamps[test_idx[0]] - timestamps[left_train[-1]]
            right_gap = None
            if right_train.size and test_idx.size:
                right_gap = timestamps[right_train[0]] - timestamps[test_idx[-1]]

            splits.append(
                PurgedCVSplit(
                    fold=fold,
                    train=train_idx,
                    test=test_idx,
                    diagnostics=PurgedCVDiagnostics(
                        test_range=test_range,
                        left_gap=left_gap,
                        right_gap=right_gap,
                    ),
                )
            )
        if not splits:
            raise ValueError("Failed to generate any purged CV splits")
        return splits

    def to_frame(self, index: Sequence[pd.Timestamp]) -> pd.DataFrame:
        """Return a dataframe summarising the purged CV splits."""

        records = [split.record() for split in self.split(index)]
        return pd.DataFrame(records)
