
"""Advanced FX data loader with timezone normalisation and DST-aware alignment."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FXDataSplit:
    """Container holding tensors for one dataset partition."""

    features: torch.Tensor
    targets: torch.Tensor
    timestamps: pd.DatetimeIndex


@dataclass(frozen=True)
class FXHorizonDataset:
    """Dataset bundle for a specific currency pair and forecasting horizon."""

    pair: str
    horizon: pd.Timedelta
    horizon_steps: int
    train: FXDataSplit
    val: FXDataSplit
    test: FXDataSplit
    feature_scaler: StandardScaler
    target_scaler: StandardScaler


@dataclass
class FXDataLoaderConfig:
    """Configuration for the FX data loader."""

    data_dir: Path
    pairs: Sequence[str]
    horizons: Sequence[object]
    feature_columns: Sequence[str]
    target_column: str
    timestamp_column: str = "timestamp"
    frequency: str = "1H"
    timezone: str = "UTC"
    lookback: int = 32
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    file_pattern: str = "{pair}.csv"

    def validate(self) -> None:
        if not np.isclose(self.train_ratio + self.val_ratio + self.test_ratio, 1.0):
            raise ValueError("Train/val/test ratios must sum to 1.0")
        if self.lookback <= 0:
            raise ValueError("lookback must be a positive integer")
        if not self.pairs:
            raise ValueError("At least one currency pair must be provided")
        if not self.horizons:
            raise ValueError("At least one forecasting horizon must be provided")
        if not self.feature_columns:
            raise ValueError("feature_columns cannot be empty")
        if not Path(self.data_dir).exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")


class FXDataLoader:
    """Read FX CSVs, normalise by train split, and emit aligned tensors."""

    def __init__(self, config: FXDataLoaderConfig):
        self.cfg = config
        self.cfg.validate()
        self._max_horizon_steps = max(self._coerce_horizon(h)[1] for h in self.cfg.horizons)

    def load(self) -> Dict[Tuple[str, pd.Timedelta], FXHorizonDataset]:
        datasets: Dict[Tuple[str, pd.Timedelta], FXHorizonDataset] = {}
        for pair in self.cfg.pairs:
            pair_df = self._load_pair_dataframe(pair)
            train_df, val_df, test_df = self._split_dataframe(pair_df)
            feature_scaler = self._fit_feature_scaler(train_df)
            for horizon in self.cfg.horizons:
                horizon_td, horizon_steps = self._coerce_horizon(horizon)
                sequences = self._prepare_sequences(
                    train_df,
                    val_df,
                    test_df,
                    feature_scaler,
                    horizon_steps,
                )
                target_scaler = self._fit_target_scaler(sequences["train"].targets)
                scaled_sequences = {
                    split: FXDataSplit(
                        features=seq.features,
                        targets=self._transform_targets(seq.targets, target_scaler),
                        timestamps=seq.timestamps,
                    )
                    for split, seq in sequences.items()
                }
                datasets[(pair, horizon_td)] = FXHorizonDataset(
                    pair=pair,
                    horizon=horizon_td,
                    horizon_steps=horizon_steps,
                    train=scaled_sequences["train"],
                    val=scaled_sequences["val"],
                    test=scaled_sequences["test"],
                    feature_scaler=feature_scaler,
                    target_scaler=target_scaler,
                )
        return datasets

    def _load_pair_dataframe(self, pair: str) -> pd.DataFrame:
        file_path = Path(self.cfg.data_dir) / self.cfg.file_pattern.format(pair=pair)
        if not file_path.exists():
            raise FileNotFoundError(f"Unable to locate dataset for {pair!r} at {file_path}")
        LOGGER.info("Loading %s", file_path)
        df = pd.read_csv(file_path)
        if self.cfg.timestamp_column not in df.columns:
            raise KeyError(
                f"Missing timestamp column '{self.cfg.timestamp_column}' in dataset {file_path}"
            )
        df = self._normalise_time_index(df)
        missing_cols = [
            col for col in list(self.cfg.feature_columns) + [self.cfg.target_column] if col not in df.columns
        ]
        if missing_cols:
            raise KeyError(f"Missing required columns for {pair}: {missing_cols}")
        df = df.dropna(subset=list(self.cfg.feature_columns) + [self.cfg.target_column])
        if df.empty:
            raise ValueError(f"Dataset for {pair} is empty after dropping missing rows")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.ffill().bfill()
        return df

    def _normalise_time_index(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = pd.to_datetime(df[self.cfg.timestamp_column], utc=False, errors="coerce")
        if ts.isnull().any():
            raise ValueError("Timestamp column contains non-parsable values")
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(
                self.cfg.timezone,
                ambiguous="infer",
                nonexistent="shift_forward",
            )
        else:
            ts = ts.dt.tz_convert(self.cfg.timezone)
        ts = ts.dt.tz_convert("UTC")
        frame = df.copy()
        frame.index = ts
        frame = frame.drop(columns=[self.cfg.timestamp_column])
        if self.cfg.frequency:
            freq = pd.tseries.frequencies.to_offset(self.cfg.frequency)
            start, end = frame.index.min(), frame.index.max()
            full_index = pd.date_range(start=start, end=end, freq=freq, tz=frame.index.tz)
            frame = frame.reindex(full_index)
        # Mirror the DST gap handling from the historical v_07 implementation by using forward/back fills.
        frame = frame.ffill()
        frame = frame.bfill()
        frame.index.name = self.cfg.timestamp_column
        return frame

    def _split_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        total = len(df)
        train_end = int(total * self.cfg.train_ratio)
        val_end = train_end + int(total * self.cfg.val_ratio)
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        if train_df.empty:
            raise ValueError("Training split is empty; adjust train_ratio or provide more data")
        return train_df, val_df, test_df

    def _fit_feature_scaler(self, train_df: pd.DataFrame) -> StandardScaler:
        scaler = StandardScaler()
        effective_train = train_df
        if self._max_horizon_steps > 0 and len(train_df) > self._max_horizon_steps:
            effective_train = train_df.iloc[:-self._max_horizon_steps]
        if effective_train.empty:
            raise ValueError("Not enough training rows to fit the feature scaler after horizon adjustment")
        scaler.fit(effective_train.loc[:, self.cfg.feature_columns])
        return scaler

    def _prepare_sequences(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_scaler: StandardScaler,
        horizon_steps: int,
    ) -> Dict[str, FXDataSplit]:
        splits = {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }
        prepared: Dict[str, FXDataSplit] = {}
        for split_name, split_df in splits.items():
            if split_df.empty:
                prepared[split_name] = FXDataSplit(
                    features=torch.empty((0, self.cfg.lookback, len(self.cfg.feature_columns))),
                    targets=torch.empty((0, 1)),
                    timestamps=pd.DatetimeIndex([], tz="UTC"),
                )
                continue
            features = feature_scaler.transform(split_df.loc[:, self.cfg.feature_columns])
            prices = split_df.loc[:, self.cfg.target_column].to_numpy(dtype=np.float64)
            timestamps = split_df.index
            windows, targets, target_ts = self._window_sequences(
                features,
                prices,
                timestamps,
                horizon_steps,
            )
            prepared[split_name] = FXDataSplit(
                features=torch.from_numpy(windows.astype(np.float32)),
                targets=torch.from_numpy(targets.astype(np.float32)),
                timestamps=target_ts,
            )
        return prepared

    def _window_sequences(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        timestamps: pd.DatetimeIndex,
        horizon_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        if horizon_steps <= 0:
            raise ValueError("horizon must be at least one step")
        num_features = features.shape[1]
        max_index = len(features) - horizon_steps
        if max_index <= self.cfg.lookback - 1:
            return (
                np.empty((0, self.cfg.lookback, num_features), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
                pd.DatetimeIndex([], tz="UTC"),
            )
        seq_list: list[np.ndarray] = []
        target_list: list[float] = []
        ts_list: list[pd.Timestamp] = []
        for end_idx in range(self.cfg.lookback - 1, max_index):
            start_idx = end_idx - self.cfg.lookback + 1
            window = features[start_idx : end_idx + 1]
            future_price = prices[end_idx + horizon_steps]
            current_price = prices[end_idx]
            if current_price <= 0 or future_price <= 0:
                continue
            seq_list.append(window)
            target_list.append(np.log(future_price) - np.log(current_price))
            ts_list.append(timestamps[end_idx + horizon_steps])
        if not seq_list:
            return (
                np.empty((0, self.cfg.lookback, num_features), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
                pd.DatetimeIndex([], tz="UTC"),
            )
        windows = np.stack(seq_list, axis=0)
        target_array = np.asarray(target_list, dtype=np.float64).reshape(-1, 1)
        ts_index = pd.DatetimeIndex(ts_list)
        if ts_index.tz is None:
            ts_index = ts_index.tz_localize(timestamps.tz or "UTC")
        else:
            ts_index = ts_index.tz_convert("UTC")
        return windows, target_array, ts_index

    def _fit_target_scaler(self, train_targets: torch.Tensor) -> StandardScaler:
        scaler = StandardScaler()
        if train_targets.numel() == 0:
            scaler.mean_ = np.zeros(1)
            scaler.scale_ = np.ones(1)
            scaler.var_ = np.ones(1)
            return scaler
        scaler.fit(train_targets.cpu().numpy())
        return scaler

    def _transform_targets(self, targets: torch.Tensor, scaler: StandardScaler) -> torch.Tensor:
        if targets.numel() == 0:
            return targets
        transformed = scaler.transform(targets.cpu().numpy())
        return torch.from_numpy(transformed.astype(np.float32))

    def _coerce_horizon(self, horizon: object) -> Tuple[pd.Timedelta, int]:
        step_td = pd.to_timedelta(self.cfg.frequency)
        if isinstance(horizon, (int, np.integer)):
            if horizon <= 0:
                raise ValueError("Integer horizons must be positive")
            horizon_steps = int(horizon)
            horizon_td = horizon_steps * step_td
        else:
            horizon_td = pd.to_timedelta(str(horizon).lower())
            if horizon_td <= pd.Timedelta(0):
                raise ValueError("Horizon timedelta must be positive")
            ratio = horizon_td / step_td
            horizon_steps = int(round(ratio))
            if not np.isclose(ratio, horizon_steps):
                raise ValueError("Horizon must align with configured frequency")
        return horizon_td, horizon_steps
