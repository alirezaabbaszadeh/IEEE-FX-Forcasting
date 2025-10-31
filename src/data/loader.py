"""Shared chronological data loader for FX time series experiments."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS: Tuple[str, ...] = ("Open", "High", "Low", "Close")
TARGET_COLUMN: str = "Close"


@dataclass(frozen=True)
class SplitBounds:
    """Half-open index bounds `[start, end)` describing a dataset partition."""

    start: int
    end: int

    @property
    def size(self) -> int:
        return max(0, self.end - self.start)


@dataclass(frozen=True)
class SplitMetadata:
    """Chronological layout of the dataset."""

    train: SplitBounds
    val: SplitBounds
    test: SplitBounds


@dataclass
class SequencedPartition:
    """Container for a time-series partition after sequencing."""

    features: np.ndarray
    target: np.ndarray

    @property
    def empty(self) -> bool:
        return self.features.size == 0 or self.target.size == 0


@dataclass
class DataLoaderArtifacts:
    """Structured artefacts emitted by :class:`ChronologicalDataLoader`."""

    train: SequencedPartition
    val: SequencedPartition
    test: SequencedPartition
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    split_metadata: SplitMetadata
    feature_columns: Tuple[str, ...]
    target_column: str


@dataclass(frozen=True)
class LegacyLoaderOptions:
    """Optional tweaks replicating historical data pre-processing quirks."""

    feature_columns: Tuple[str, ...] = FEATURE_COLUMNS
    target_column: str = TARGET_COLUMN
    timestamp_column: str | None = None
    timestamp_format: str | None = None
    sort_by_timestamp: bool = False
    drop_duplicate_timestamps: bool = False
    duplicate_keep: str = "first"
    dropna_columns: Tuple[str, ...] | None = None

    def with_overrides(self, **overrides: Any) -> "LegacyLoaderOptions":
        """Return a copy with selected fields replaced."""

        valid_fields = {field.name for field in fields(self)}
        invalid = set(overrides) - valid_fields
        if invalid:
            invalid_list = ", ".join(sorted(invalid))
            raise TypeError(f"Invalid loader option override(s): {invalid_list}")
        return replace(self, **overrides)


class ChronologicalDataLoader:
    """Chronological DataLoader matching the legacy `v_*/DataLoader.py` contract."""

    def __init__(
        self,
        file_path: str,
        *,
        time_steps: int = 3,
        train_ratio: float = 0.94,
        val_ratio: float = 0.03,
        test_ratio: float = 0.03,
        options: LegacyLoaderOptions | None = None,
        **option_overrides: Any,
    ) -> None:
        self.file_path = Path(file_path)
        self.time_steps = int(time_steps)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)

        base_options = options or LegacyLoaderOptions()
        self.options = base_options.with_overrides(**option_overrides)
        self.feature_columns = tuple(self.options.feature_columns)
        self.target_column = self.options.target_column

        if not np.isclose(self.train_ratio + self.val_ratio + self.test_ratio, 1.0):
            raise ValueError("Split ratios must sum to 1.0")

    def load(self) -> DataLoaderArtifacts:
        frame = self._load_frame()
        if frame.empty:
            raise ValueError("Loaded data is empty.")

        split_metadata = self._compute_split_metadata(len(frame))
        train_df = frame.iloc[split_metadata.train.start : split_metadata.train.end]
        val_df = frame.iloc[split_metadata.val.start : split_metadata.val.end]
        test_df = frame.iloc[split_metadata.test.start : split_metadata.test.end]

        X_train_raw, y_train_raw = self._extract(train_df)
        X_val_raw, y_val_raw = self._extract(val_df)
        X_test_raw, y_test_raw = self._extract(test_df)

        if X_train_raw.shape[0] == 0:
            raise ValueError("Training dataset is empty after splitting.")

        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        feature_scaler.fit(X_train_raw)
        target_scaler.fit(y_train_raw)

        X_train_scaled = feature_scaler.transform(X_train_raw)
        y_train_scaled = target_scaler.transform(y_train_raw)
        X_val_scaled, y_val_scaled = self._transform_optional(
            X_val_raw, y_val_raw, feature_scaler, target_scaler
        )
        X_test_scaled, y_test_scaled = self._transform_optional(
            X_test_raw, y_test_raw, feature_scaler, target_scaler
        )

        num_features = len(self.feature_columns)
        train_partition = self._sequence_partition(X_train_scaled, y_train_scaled, num_features)
        val_partition = self._sequence_partition(X_val_scaled, y_val_scaled, num_features)
        test_partition = self._sequence_partition(X_test_scaled, y_test_scaled, num_features)

        return DataLoaderArtifacts(
            train=train_partition,
            val=val_partition,
            test=test_partition,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            split_metadata=split_metadata,
            feature_columns=self.feature_columns,
            target_column=self.target_column,
        )

    def get_data(self) -> Tuple[np.ndarray, ...]:
        """Legacy-compatible alias returning arrays plus the target scaler."""

        artifacts = self.load()
        return (
            artifacts.train.features,
            artifacts.val.features,
            artifacts.test.features,
            artifacts.train.target,
            artifacts.val.target,
            artifacts.test.target,
            artifacts.target_scaler,
        )

    def create_sequences(
        self, X_data: np.ndarray, y_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Expose the sliding-window sequence builder for legacy callers."""

        num_features = X_data.shape[1] if X_data.ndim == 2 else len(self.feature_columns)
        partition = self._sequence_partition(X_data, y_data, num_features)
        return partition.features, partition.target

    def _load_frame(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        frame = pd.read_csv(self.file_path)
        frame = frame.dropna()

        options = self.options
        dropna_columns: Tuple[str, ...] | None = options.dropna_columns
        if dropna_columns:
            frame = frame.dropna(subset=list(dropna_columns))

        if options.timestamp_column:
            timestamp_series = pd.to_datetime(
                frame[options.timestamp_column],
                format=options.timestamp_format,
                errors="coerce",
            )
            frame = frame.assign(**{options.timestamp_column: timestamp_series})
            frame = frame.dropna(subset=[options.timestamp_column])
            if options.sort_by_timestamp:
                frame = frame.sort_values(options.timestamp_column, kind="mergesort")
            if options.drop_duplicate_timestamps:
                frame = frame.drop_duplicates(
                    subset=[options.timestamp_column], keep=options.duplicate_keep
                )

        return frame.reset_index(drop=True)

    def _compute_split_metadata(self, total_rows: int) -> SplitMetadata:
        train_end = int(self.train_ratio * total_rows)
        val_end = int((self.train_ratio + self.val_ratio) * total_rows)
        return SplitMetadata(
            train=SplitBounds(0, train_end),
            val=SplitBounds(train_end, val_end),
            test=SplitBounds(val_end, total_rows),
        )

    def _extract(self, frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if frame.empty:
            return np.empty((0, len(self.feature_columns))), np.empty((0, 1))

        X = frame.loc[:, self.feature_columns].values
        y = frame.loc[:, self.target_column].values.reshape(-1, 1)
        return X, y

    def _transform_optional(
        self,
        X_raw: np.ndarray,
        y_raw: np.ndarray,
        feature_scaler: StandardScaler,
        target_scaler: StandardScaler,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if X_raw.shape[0] == 0:
            return np.empty((0, len(self.feature_columns))), np.empty((0, 1))
        X_scaled = feature_scaler.transform(X_raw)
        y_scaled = target_scaler.transform(y_raw)
        return X_scaled, y_scaled

    def _sequence_partition(
        self, X: np.ndarray, y: np.ndarray, num_features: int
    ) -> SequencedPartition:
        if X.shape[0] <= self.time_steps:
            empty_features = np.empty((0, self.time_steps, num_features))
            empty_target = np.empty((0, 1))
            return SequencedPartition(empty_features, empty_target)

        sequences = []
        targets = []
        for idx in range(self.time_steps, X.shape[0]):
            sequences.append(X[idx - self.time_steps : idx])
            targets.append(y[idx])
        return SequencedPartition(np.array(sequences), np.array(targets))


__all__ = [
    "ChronologicalDataLoader",
    "DataLoaderArtifacts",
    "LegacyLoaderOptions",
    "SequencedPartition",
    "SplitMetadata",
    "SplitBounds",
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
]
