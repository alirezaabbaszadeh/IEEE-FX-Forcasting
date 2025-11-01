"""Walk-forward dataset construction utilities."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Dict

import numpy as np
import pandas as pd
import pytz
from sklearn.preprocessing import StandardScaler

from src.data.dataset import DataConfig, SequenceDataset, WindowedData
from src.eval.scheduler import WalkForwardConfig, WalkForwardScheduler

LOGGER = logging.getLogger(__name__)


class WalkForwardSplitter:
    """Generate sequential walk-forward datasets per pair and horizon."""

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        wf = cfg.walkforward
        if wf is None:  # pragma: no cover - guarded by DataConfig.validate
            raise ValueError("walkforward configuration is required")
        self.scheduler = WalkForwardScheduler(
            WalkForwardConfig(
                train_size=wf.train,
                val_size=wf.val,
                test_size=wf.test,
                step=wf.step,
                embargo=wf.embargo,
            )
        )

    def __call__(self, df: pd.DataFrame) -> Dict[tuple[str, pd.Timedelta, int], WindowedData]:
        frame = self._normalise_dataframe(df)
        outputs: Dict[tuple[str, pd.Timedelta, int], WindowedData] = {}

        for pair in self.cfg.pairs:
            pair_frame = frame[frame[self.cfg.pair_column] == pair]
            if pair_frame.empty:
                raise ValueError(f"No rows available for pair {pair!r}")

            pair_frame = pair_frame.set_index(self.cfg.timestamp_column)
            pair_frame = pair_frame.sort_index()
            pair_frame = pair_frame[~pair_frame.index.duplicated(keep="last")]

            freq_td = self._infer_frequency(pair_frame.index)
            windows = self.scheduler.generate(pair_frame.index)

            for raw_horizon in self.cfg.horizons:
                horizon_td, horizon_steps = self._resolve_horizon(raw_horizon, freq_td)
                windowed = self._build_pair_windows(
                    pair, pair_frame, windows, horizon_td, horizon_steps
                )
                for window_id, data in windowed.items():
                    outputs[(pair, horizon_td, window_id)] = data

        if not outputs:
            raise ValueError("Failed to construct any walk-forward datasets")
        return outputs

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _normalise_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        raw_timestamps = df[self.cfg.timestamp_column]
        timestamps = pd.to_datetime(raw_timestamps, errors="coerce")
        if timestamps.isnull().any():
            raise ValueError("Timestamp column contains non-parsable values")

        source_tz = pytz.timezone(self.cfg.timezone.source)
        target_tz = pytz.timezone(self.cfg.timezone.normalise_to)

        if timestamps.dtype == "object":
            timestamps = pd.to_datetime(raw_timestamps, errors="coerce", utc=True)
            if timestamps.isnull().any():
                raise ValueError("Timestamp column contains non-parsable values")
            timestamps = timestamps.dt.tz_convert(source_tz)
        elif timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize(
                source_tz,
                ambiguous="infer",
                nonexistent="shift_forward",
            )
        else:
            timestamps = timestamps.dt.tz_convert(source_tz)

        timestamps = timestamps.dt.tz_convert(target_tz)
        timestamps = timestamps.dt.tz_localize(None)
        timestamps = timestamps.dt.tz_localize(target_tz)

        frame = df.copy()
        frame[self.cfg.timestamp_column] = timestamps
        frame = frame.sort_values(by=self.cfg.timestamp_column)
        return frame

    def _infer_frequency(self, index: pd.DatetimeIndex) -> pd.Timedelta:
        freq = pd.infer_freq(index)
        if freq is not None:
            if isinstance(freq, str) and not any(ch.isdigit() for ch in freq):
                freq = f"1{freq}"
            return pd.to_timedelta(freq)

        diffs = index.to_series().diff().dropna()
        if diffs.empty:
            raise ValueError("Cannot infer frequency from a single timestamp")
        freq_td = diffs.mode().iloc[0]
        if not isinstance(freq_td, pd.Timedelta):
            freq_td = pd.to_timedelta(freq_td)
        return freq_td

    def _resolve_horizon(self, horizon: object, freq_td: pd.Timedelta) -> tuple[pd.Timedelta, int]:
        if isinstance(horizon, str):
            horizon_td = pd.to_timedelta(horizon)
            steps_float = horizon_td / freq_td
            steps = int(round(float(steps_float)))
        elif isinstance(horizon, (int, np.integer)):
            steps = int(horizon)
            horizon_td = freq_td * steps
        else:
            raise TypeError(f"Unsupported horizon type: {type(horizon)!r}")

        if steps <= 0:
            raise ValueError("Horizon must be at least one step")

        if isinstance(horizon, str):
            if not np.isclose(steps_float, steps):
                raise ValueError(
                    f"Horizon {horizon!r} is not an integer multiple of the inferred frequency {freq_td}"
                )

        return horizon_td, steps

    def _build_pair_windows(
        self,
        pair: str,
        frame: pd.DataFrame,
        windows: list,
        horizon_td: pd.Timedelta,
        horizon_steps: int,
    ) -> Dict[int, WindowedData]:
        feature_columns = list(self.cfg.feature_columns)
        target_column = self.cfg.target_column

        outputs: Dict[int, WindowedData] = {}

        for window_id, window in enumerate(windows):
            train_df = frame.iloc[window.train]
            val_df = frame.iloc[window.val]
            test_df = frame.iloc[window.test]

            if train_df.empty:
                raise ValueError("Training window is empty after filtering")

            feature_scaler = StandardScaler().fit(train_df.loc[:, feature_columns])
            target_scaler = StandardScaler().fit(train_df[[target_column]])

            train_dataset = self._build_partition_dataset(
                train_df, feature_scaler, target_scaler, horizon_steps
            )
            val_dataset = self._build_partition_dataset(
                val_df, feature_scaler, target_scaler, horizon_steps
            )
            test_dataset = self._build_partition_dataset(
                test_df, feature_scaler, target_scaler, horizon_steps
            )

            metadata = {
                "pair": pair,
                "horizon": horizon_td,
                "horizon_steps": horizon_steps,
                "window_id": window_id,
                "embargo": self.cfg.walkforward.embargo if self.cfg.walkforward else 0,
                "timezone": self.cfg.timezone.normalise_to,
                "calendar": asdict(self.cfg.calendar),
                "train_index": train_df.index.copy(),
                "val_index": val_df.index.copy(),
                "test_index": test_df.index.copy(),
            }

            outputs[window_id] = WindowedData(
                train=train_dataset,
                val=val_dataset,
                test=test_dataset,
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
                metadata=metadata,
            )

        return outputs

    def _build_partition_dataset(
        self,
        df: pd.DataFrame,
        feature_scaler: StandardScaler,
        target_scaler: StandardScaler,
        horizon_steps: int,
    ) -> SequenceDataset:
        feature_columns = list(self.cfg.feature_columns)
        target_column = self.cfg.target_column

        if df.empty:
            num_features = len(feature_columns)
            return SequenceDataset(
                np.empty((0, self.cfg.time_steps, num_features), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
            )

        scaled_features = feature_scaler.transform(df.loc[:, feature_columns]).astype(np.float32)
        scaled_targets = target_scaler.transform(df[[target_column]]).astype(np.float32).reshape(-1)

        sequences, targets = self._create_sequences(scaled_features, scaled_targets, horizon_steps)
        return SequenceDataset(sequences, targets)

    def _create_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        horizon_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        lookback = self.cfg.time_steps
        total = len(features)
        limit = total - lookback - horizon_steps + 1
        num_features = features.shape[1]

        if limit <= 0:
            return (
                np.empty((0, lookback, num_features), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
            )

        sequences = np.empty((limit, lookback, num_features), dtype=np.float32)
        target_array = np.empty((limit, 1), dtype=np.float32)

        for idx in range(limit):
            start = idx
            end = start + lookback
            target_idx = end + horizon_steps - 1
            sequences[idx] = features[start:end]
            target_array[idx, 0] = targets[target_idx]

        return sequences, target_array
