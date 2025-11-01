from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import (
    CalendarConfig,
    DataConfig,
    TimezoneConfig,
    WalkForwardSettings,
    prepare_datasets,
)


def _write_fx_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.write_text(frame.to_csv(index=False))
    return path


def _base_config(csv_path: Path, train: int, val: int, test: int, step: int, embargo: int) -> DataConfig:
    return DataConfig(
        csv_path=csv_path,
        feature_columns=["open", "high", "low", "close"],
        target_column="close",
        timestamp_column="timestamp",
        pair_column="pair",
        pairs=["EURUSD"],
        horizons=[1],
        time_steps=4,
        batch_size=8,
        num_workers=0,
        shuffle_train=False,
        timezone=TimezoneConfig(source="America/New_York", normalise_to="UTC"),
        calendar=CalendarConfig(primary="fx_primary", fallback="fx_backup"),
        walkforward=WalkForwardSettings(train=train, val=val, test=test, step=step, embargo=embargo),
    )


@pytest.fixture
def hourly_dataframe() -> pd.DataFrame:
    periods = 80
    timestamps = pd.date_range("2021-01-01", periods=periods, freq="1H", tz="America/New_York")
    prices = np.linspace(1.0, 3.0, periods)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "pair": "EURUSD",
            "open": prices,
            "high": prices + 0.1,
            "low": prices - 0.1,
            "close": prices + 0.05,
        }
    )
    return frame


def test_embargo_enforced_between_partitions(tmp_path: Path, hourly_dataframe: pd.DataFrame) -> None:
    csv_path = _write_fx_csv(tmp_path / "embargo.csv", hourly_dataframe)
    cfg = _base_config(csv_path, train=30, val=10, test=10, step=10, embargo=2)

    datasets = prepare_datasets(cfg)
    assert datasets

    freq_td = pd.to_timedelta("1H")
    for key, window in datasets.items():
        assert len(key) == 3
        metadata = window.metadata
        train_index = metadata["train_index"]
        val_index = metadata["val_index"]
        test_index = metadata["test_index"]

        if len(val_index) > 0:
            assert (val_index[0] - train_index[-1]) >= freq_td * (cfg.walkforward.embargo + 1)
        if len(test_index) > 0:
            assert (test_index[0] - val_index[-1]) >= freq_td * (cfg.walkforward.embargo + 1)

        assert metadata["calendar"]["primary"] == "fx_primary"
        assert metadata["timezone"] == "UTC"


def test_timezone_normalisation_handles_dst(tmp_path: Path) -> None:
    periods = 96
    timestamps = pd.date_range("2020-03-07", periods=periods, freq="30min", tz="America/New_York")
    prices = np.linspace(10.0, 20.0, periods)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "pair": "EURUSD",
            "open": prices,
            "high": prices + 0.2,
            "low": prices - 0.2,
            "close": prices + 0.1,
        }
    )
    csv_path = _write_fx_csv(tmp_path / "dst.csv", frame)
    cfg = _base_config(csv_path, train=40, val=20, test=20, step=20, embargo=1)

    datasets = prepare_datasets(cfg)
    assert datasets

    for window in datasets.values():
        for split_name in ("train_index", "val_index", "test_index"):
            index = window.metadata[split_name]
            if len(index) == 0:
                continue
            assert index.tz is not None
            assert index.tz.zone == "UTC"
            diffs = index.to_series().diff().dropna()
            if not diffs.empty:
                expected = diffs.mode().iloc[0]
                assert (diffs == expected).all()


def test_scalers_use_train_only(tmp_path: Path, hourly_dataframe: pd.DataFrame) -> None:
    shifted = hourly_dataframe.copy()
    shifted.loc[hourly_dataframe.index >= 30, ["open", "high", "low", "close"]] += 5.0
    shifted.loc[hourly_dataframe.index >= 50, ["open", "high", "low", "close"]] += 7.0
    csv_path = _write_fx_csv(tmp_path / "scalers.csv", shifted)
    cfg = _base_config(csv_path, train=24, val=12, test=12, step=12, embargo=1)

    datasets = prepare_datasets(cfg)
    assert len(datasets) >= 2

    source_df = shifted.set_index("timestamp").tz_convert("UTC")
    scaler_means: set[tuple[float, ...]] = set()

    for (_, _, window_id), window in datasets.items():
        metadata = window.metadata
        train_index = metadata["train_index"]
        val_index = metadata["val_index"]
        test_index = metadata["test_index"]

        train_df = source_df.loc[train_index]
        expected_mean = train_df.loc[:, cfg.feature_columns].mean().to_numpy()
        np.testing.assert_allclose(window.feature_scaler.mean_, expected_mean, atol=1e-6)

        scaler_means.add(tuple(window.feature_scaler.mean_.round(6)))

        if len(val_index) > 0:
            val_df = source_df.loc[val_index]
            val_mean = val_df.loc[:, cfg.feature_columns].mean().to_numpy()
            assert np.any(np.abs(val_mean - expected_mean) > 1e-3)

        if len(test_index) > 0:
            test_df = source_df.loc[test_index]
            test_mean = test_df.loc[:, cfg.feature_columns].mean().to_numpy()
            assert np.any(np.abs(test_mean - expected_mean) > 1e-3)

    assert len(scaler_means) >= 2
