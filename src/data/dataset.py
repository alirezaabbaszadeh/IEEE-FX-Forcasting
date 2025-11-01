"""Data loading utilities for the consolidated `src` package."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration required to materialise the training datasets."""

    csv_path: Path
    feature_columns: Sequence[str]
    target_column: str
    pairs: Sequence[str] = ()
    horizons: Sequence[int] = (1,)
    time_steps: int = 32
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    batch_size: int = 64
    num_workers: int = 0
    shuffle_train: bool = True

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(
                "Train/val/test ratios must sum to 1.0. "
                f"Received {self.train_ratio}, {self.val_ratio}, {self.test_ratio}."
            )
        if self.time_steps <= 0:
            raise ValueError("time_steps must be a positive integer")

        if not Path(self.csv_path).exists():
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")


class SequenceDataset(Dataset):
    """Windowed time-series dataset compatible with PyTorch dataloaders."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        if sequences.shape[0] != targets.shape[0]:
            raise ValueError("Number of sequences and targets must match")
        self._sequences = torch.from_numpy(sequences.astype(np.float32))
        self._targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._sequences.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover - trivial
        return self._sequences[index], self._targets[index]


@dataclass
class PreparedData:
    train: SequenceDataset
    val: SequenceDataset
    test: SequenceDataset
    feature_scaler: StandardScaler
    target_scaler: StandardScaler


def create_dataloaders(data: PreparedData, cfg: DataConfig) -> dict[str, DataLoader]:
    """Materialise PyTorch dataloaders with consistent settings."""

    return {
        "train": DataLoader(
            data.train,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle_train,
            num_workers=cfg.num_workers,
        ),
        "val": DataLoader(
            data.val,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        ),
        "test": DataLoader(
            data.test,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        ),
    }


def load_dataframe(csv_path: Path, feature_columns: Iterable[str], target_column: str) -> pd.DataFrame:
    """Load and validate the raw dataframe."""

    LOGGER.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)
    missing = [col for col in list(feature_columns) + [target_column] if col not in df.columns]
    if missing:
        raise KeyError(f"Columns missing from dataset: {missing}")
    df = df.dropna(subset=list(feature_columns) + [target_column])
    LOGGER.debug("Loaded dataframe shape: %s", df.shape)
    return df


def _split_indices(length: int, cfg: DataConfig) -> Tuple[slice, slice, slice]:
    train_end = int(length * cfg.train_ratio)
    val_end = train_end + int(length * cfg.val_ratio)
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, length)


def _create_sequences(values: np.ndarray, targets: np.ndarray, time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(values) <= time_steps:
        LOGGER.warning(
            "Not enough rows (%s) to build sequences with %s time steps. Returning empty arrays.",
            len(values),
            time_steps,
        )
        num_features = values.shape[1]
        return np.empty((0, time_steps, num_features), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for start in range(len(values) - time_steps):
        end = start + time_steps
        xs.append(values[start:end])
        ys.append(targets[end])

    sequences = np.stack(xs, axis=0)
    y_array = np.stack(ys, axis=0)
    return sequences.astype(np.float32), y_array.astype(np.float32)


def prepare_datasets(cfg: DataConfig) -> PreparedData:
    """Load a CSV file and return scaled `SequenceDataset` objects."""

    cfg.validate()
    df = load_dataframe(cfg.csv_path, cfg.feature_columns, cfg.target_column)

    feature_values = df.loc[:, cfg.feature_columns].to_numpy(dtype=np.float32)
    target_values = df.loc[:, cfg.target_column].to_numpy(dtype=np.float32).reshape(-1, 1)

    train_slice, val_slice, test_slice = _split_indices(len(df), cfg)

    x_train_raw = feature_values[train_slice]
    y_train_raw = target_values[train_slice]

    feature_scaler = StandardScaler().fit(x_train_raw)
    target_scaler = StandardScaler().fit(y_train_raw)

    def scale_and_window(data_slice: slice) -> Tuple[np.ndarray, np.ndarray]:
        x_scaled = feature_scaler.transform(feature_values[data_slice])
        y_scaled = target_scaler.transform(target_values[data_slice])
        return _create_sequences(x_scaled, y_scaled, cfg.time_steps)

    train_seq, train_targets = scale_and_window(train_slice)
    val_seq, val_targets = scale_and_window(val_slice)
    test_seq, test_targets = scale_and_window(test_slice)

    LOGGER.info(
        "Prepared datasets - train: %s, val: %s, test: %s",
        train_seq.shape,
        val_seq.shape,
        test_seq.shape,
    )

    return PreparedData(
        train=SequenceDataset(train_seq, train_targets),
        val=SequenceDataset(val_seq, val_targets),
        test=SequenceDataset(test_seq, test_targets),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )
