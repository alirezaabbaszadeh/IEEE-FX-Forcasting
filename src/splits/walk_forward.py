"""Walk-forward split utilities with embargo diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

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


@dataclass(frozen=True)
class WalkForwardDiagnostics:
    """Metadata describing embargo gaps for a walk-forward split."""

    train_range: tuple[pd.Timestamp, pd.Timestamp] | None
    val_range: tuple[pd.Timestamp, pd.Timestamp] | None
    test_range: tuple[pd.Timestamp, pd.Timestamp] | None
    embargo_gap_train_val: pd.Timedelta | None
    embargo_gap_val_test: pd.Timedelta | None
    overlaps_previous_window: bool

    def to_metadata(self) -> dict[str, object]:
        """Serialise diagnostics for inclusion in dataset metadata."""

        def _range_payload(value: tuple[pd.Timestamp, pd.Timestamp] | None) -> dict[str, str] | None:
            if value is None:
                return None
            start, end = value
            return {"start": start.isoformat(), "end": end.isoformat()}

        payload: dict[str, object] = {
            "train_range": _range_payload(self.train_range),
            "val_range": _range_payload(self.val_range),
            "test_range": _range_payload(self.test_range),
            "embargo_gap_train_val": self.embargo_gap_train_val,
            "embargo_gap_val_test": self.embargo_gap_val_test,
            "overlaps_previous_window": self.overlaps_previous_window,
        }
        return payload


@dataclass(frozen=True)
class WalkForwardSplit:
    """Container bundling split indices with diagnostics."""

    window_id: int
    window: WalkForwardWindow
    diagnostics: WalkForwardDiagnostics

    def records(
        self,
        *,
        pair: str | None = None,
        horizon: object | None = None,
    ) -> list[dict[str, object]]:
        """Return serialisable audit rows for this split."""

        ranges = {
            "train": self.diagnostics.train_range,
            "val": self.diagnostics.val_range,
            "test": self.diagnostics.test_range,
        }
        sizes = {
            "train": int(self.window.train.size),
            "val": int(self.window.val.size),
            "test": int(self.window.test.size),
        }
        gaps = {
            "train": None,
            "val": self.diagnostics.embargo_gap_train_val,
            "test": self.diagnostics.embargo_gap_val_test,
        }
        rows: list[dict[str, object]] = []
        for split_name in ("train", "val", "test"):
            start_end = ranges[split_name]
            start = start_end[0].isoformat() if start_end else None
            end = start_end[1].isoformat() if start_end else None
            gap = gaps[split_name]
            rows.append(
                {
                    "pair": pair,
                    "horizon": str(horizon) if horizon is not None else None,
                    "window_id": self.window_id,
                    "split": split_name,
                    "size": sizes[split_name],
                    "start": start,
                    "end": end,
                    "embargo_gap_before": str(gap) if gap is not None else None,
                    "overlaps_previous_window": self.diagnostics.overlaps_previous_window,
                }
            )
        return rows


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


class WalkForwardSplitter:
    """Materialise walk-forward splits along with embargo diagnostics."""

    def __init__(self, config: WalkForwardConfig):
        self.scheduler = WalkForwardScheduler(config)

    def split(self, index: Sequence[pd.Timestamp]) -> list[WalkForwardSplit]:
        windows = self.scheduler.generate(index)
        diagnostics = _build_diagnostics(index, windows)
        return [
            WalkForwardSplit(window_id=idx, window=window, diagnostics=diagnostics[idx])
            for idx, window in enumerate(windows)
        ]

    def to_frame(
        self,
        index: Sequence[pd.Timestamp],
        *,
        pair: str | None = None,
        horizon: object | None = None,
    ) -> pd.DataFrame:
        """Convert split diagnostics to a tabular representation."""

        rows: list[dict[str, object]] = []
        for split in self.split(index):
            rows.extend(split.records(pair=pair, horizon=horizon))
        return pd.DataFrame(rows)


def _build_diagnostics(
    index: Sequence[pd.Timestamp],
    windows: Iterable[WalkForwardWindow],
) -> list[WalkForwardDiagnostics]:
    timestamp_index = pd.DatetimeIndex(index)
    diagnostics: list[WalkForwardDiagnostics] = []
    previous_test_end: pd.Timestamp | None = None

    def _range_payload(idx: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if idx.empty:
            return None
        return idx[0], idx[-1]

    def _gap(start_idx: pd.DatetimeIndex, end_idx: pd.DatetimeIndex) -> pd.Timedelta | None:
        if start_idx.empty or end_idx.empty:
            return None
        return start_idx[0] - end_idx[-1]

    for window in windows:
        resolved = window.resolve(timestamp_index)
        train_idx = resolved["train"]
        val_idx = resolved["val"]
        test_idx = resolved["test"]

        gap_train_val = _gap(val_idx, train_idx)
        gap_val_test = _gap(test_idx, val_idx)

        overlaps_previous = False
        first_available = None
        if not train_idx.empty:
            first_available = train_idx[0]
        elif not val_idx.empty:
            first_available = val_idx[0]
        elif not test_idx.empty:
            first_available = test_idx[0]

        if previous_test_end is not None and first_available is not None:
            overlaps_previous = first_available <= previous_test_end

        diagnostics.append(
            WalkForwardDiagnostics(
                train_range=_range_payload(train_idx),
                val_range=_range_payload(val_idx),
                test_range=_range_payload(test_idx),
                embargo_gap_train_val=gap_train_val,
                embargo_gap_val_test=gap_val_test,
                overlaps_previous_window=overlaps_previous,
            )
        )

        if not test_idx.empty:
            previous_test_end = test_idx[-1]
        elif not val_idx.empty:
            previous_test_end = val_idx[-1]
        elif not train_idx.empty:
            previous_test_end = train_idx[-1]

    return diagnostics
