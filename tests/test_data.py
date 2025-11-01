
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.loader import FXDataLoader, FXDataLoaderConfig
from src.eval.scheduler import WalkForwardConfig, WalkForwardScheduler


@pytest.fixture
def synthetic_fx_dir(tmp_path: Path) -> Path:
    periods = 120
    freq = "30min"
    timezone = "America/New_York"
    timestamps = (
        pd.date_range("2020-03-07 00:00", periods=periods, freq=freq, tz=timezone)
        .tz_convert(timezone)
        .tz_localize(None)
    )

    def base_value(index: int) -> float:
        if index < 60:
            return 1.0 + 0.02 * index
        if index < 90:
            return 10.0 + 0.05 * (index - 60)
        return 20.0 + 0.1 * (index - 90)

    values = np.array([base_value(i) for i in range(periods)], dtype=np.float64)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": values,
            "high": values + 0.1,
            "low": values - 0.1,
            "close": values + 0.05,
        }
    )
    frame.loc[5, ["open", "close"]] = np.nan
    frame.loc[70, ["open", "close"]] = np.nan
    csv_path = tmp_path / "EURUSD.csv"
    frame.to_csv(csv_path, index=False)
    return tmp_path


@pytest.fixture
def loader_config(synthetic_fx_dir: Path) -> FXDataLoaderConfig:
    return FXDataLoaderConfig(
        data_dir=synthetic_fx_dir,
        pairs=["EURUSD"],
        horizons=[1, "1H"],
        feature_columns=["open", "high", "low", "close"],
        target_column="close",
        timestamp_column="timestamp",
        frequency="30min",
        timezone="America/New_York",
        lookback=8,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )


def test_loader_normalization_isolated(loader_config: FXDataLoaderConfig) -> None:
    loader = FXDataLoader(loader_config)
    datasets = loader.load()
    assert len(datasets) == len(loader_config.horizons)
    for dataset in datasets.values():
        train_features = dataset.train.features.numpy()
        assert train_features.size > 0
        df = loader._load_pair_dataframe("EURUSD")
        train_df, val_df, test_df = loader._split_dataframe(df)
        max_steps = max(loader._coerce_horizon(h)[1] for h in loader_config.horizons)
        effective_train = train_df.iloc[:-max_steps] if len(train_df) > max_steps else train_df
        expected_mean = effective_train.loc[:, loader_config.feature_columns].mean().to_numpy()
        expected_std = effective_train.loc[:, loader_config.feature_columns].std(ddof=0).to_numpy()
        np.testing.assert_allclose(dataset.feature_scaler.mean_, expected_mean, atol=1e-6)
        np.testing.assert_allclose(dataset.feature_scaler.scale_, expected_std, atol=1e-6)
        if not val_df.empty:
            val_mean = val_df.loc[:, loader_config.feature_columns].mean().to_numpy()
            assert np.any(np.abs(val_mean - dataset.feature_scaler.mean_) > 5e-2)
        if not test_df.empty:
            test_mean = test_df.loc[:, loader_config.feature_columns].mean().to_numpy()
            assert np.any(np.abs(test_mean - dataset.feature_scaler.mean_) > 5e-2)
        train_targets = dataset.train.targets.numpy()
        if train_targets.size > 0:
            np.testing.assert_allclose(train_targets.mean(), 0.0, atol=1e-2)
            np.testing.assert_allclose(train_targets.std(), 1.0, atol=1e-2)


def test_loader_handles_missing_data_and_timezone(loader_config: FXDataLoaderConfig) -> None:
    loader = FXDataLoader(loader_config)
    datasets = loader.load()
    pair_key = next(iter(datasets))
    dataset = datasets[pair_key]
    for split in (dataset.train, dataset.val, dataset.test):
        if split.features.numel() > 0:
            assert not torch.isnan(split.features).any()
        if split.targets.numel() > 0:
            assert not torch.isnan(split.targets).any()
        if len(split.timestamps) > 0:
            assert split.timestamps.tz is not None
            freq = pd.to_timedelta(loader_config.frequency)
            diffs = split.timestamps.to_series().diff().dropna()
            if not diffs.empty:
                assert (diffs == freq).all()


def test_walk_forward_scheduler_embargo() -> None:
    index = pd.date_range("2021-01-01", periods=80, freq="h")
    config = WalkForwardConfig(train_size=20, val_size=8, test_size=8, embargo=2, step=8)
    scheduler = WalkForwardScheduler(config)
    windows = scheduler.generate(index)
    assert windows
    expected_gap = config.embargo + 1
    for window in windows:
        assert window.train.size == config.train_size
        assert window.val.size == config.val_size
        assert window.test.size == config.test_size
        if window.val.size:
            assert int(window.val[0]) - int(window.train[-1]) >= expected_gap
        if window.test.size:
            assert int(window.test[0]) - int(window.val[-1]) >= expected_gap
    split_dicts = scheduler.splits_for(index)
    assert len(split_dicts) == len(windows)
    for mapping, window in zip(split_dicts, windows, strict=True):
        assert np.array_equal(mapping["train"], window.train)
        assert np.array_equal(mapping["val"], window.val)
        assert np.array_equal(mapping["test"], window.test)
