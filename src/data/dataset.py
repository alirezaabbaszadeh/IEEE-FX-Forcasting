"""Data loading utilities for the consolidated `src` package."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger(__name__)


@dataclass
class TimezoneConfig:
    """Timezone settings for normalising timestamp columns."""

    source: str = "UTC"
    normalise_to: str = "UTC"


@dataclass
class CalendarConfig:
    """Information about the trading calendar sources used for alignment."""

    primary: str | None = None
    fallback: str | None = None


@dataclass
class WalkForwardSettings:
    """Hyperparameters governing walk-forward window construction."""

    train: int
    val: int
    test: int
    step: int | None = None
    embargo: int = 0

    def validate(self) -> None:
        if self.train <= 0 or self.val <= 0 or self.test <= 0:
            raise ValueError("train, val and test window sizes must be positive integers")
        if self.step is not None and self.step <= 0:
            raise ValueError("step must be a positive integer when provided")
        if self.embargo < 0:
            raise ValueError("embargo cannot be negative")


@dataclass
class DataConfig:
    """Configuration required to materialise the training datasets."""

    csv_path: Path
    feature_columns: Sequence[str]
    target_column: str
    timestamp_column: str = "timestamp"
    pair_column: str = "pair"
    pairs: Sequence[str] = ()
    horizons: Sequence[int] = (1,)
    time_steps: int = 32
    batch_size: int = 64
    num_workers: int = 0
    shuffle_train: bool = True
    timezone: TimezoneConfig = field(default_factory=TimezoneConfig)
    calendar: CalendarConfig = field(default_factory=CalendarConfig)
    walkforward: WalkForwardSettings | None = None

    def validate(self) -> None:
        if self.time_steps <= 0:
            raise ValueError("time_steps must be a positive integer")
        if not self.pairs:
            raise ValueError("At least one trading pair must be specified")
        if not self.horizons:
            raise ValueError("At least one forecasting horizon must be specified")

        if not Path(self.csv_path).exists():
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")

        if self.walkforward is None:
            raise ValueError("walkforward configuration must be provided")
        self.walkforward.validate()


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
class WindowedData:
    train: SequenceDataset
    val: SequenceDataset
    test: SequenceDataset
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    metadata: dict[str, object]


def create_dataloaders(data: WindowedData, cfg: DataConfig) -> dict[str, DataLoader]:
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


def load_dataframe(cfg: DataConfig) -> pd.DataFrame:
    """Load and validate the raw dataframe."""

    LOGGER.info("Loading data from %s", cfg.csv_path)
    df = pd.read_csv(cfg.csv_path)
    required_columns = set(cfg.feature_columns) | {cfg.target_column, cfg.timestamp_column, cfg.pair_column}
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Columns missing from dataset: {missing}")
    df = df.dropna(subset=list(required_columns))
    LOGGER.debug("Loaded dataframe shape: %s", df.shape)
    return df


def prepare_datasets(cfg: DataConfig) -> Dict[tuple[str, object, int], WindowedData]:
    """Load CSV data and generate walk-forward window datasets keyed by metadata."""

    from src.data.walkforward import WalkForwardSplitter

    cfg.validate()
    df = load_dataframe(cfg)

    splitter = WalkForwardSplitter(cfg)
    return splitter(df)
