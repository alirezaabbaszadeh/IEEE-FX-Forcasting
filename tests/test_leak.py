"""Leak guard regression tests for walk-forward dataset preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import (
    DataConfig,
    TimezoneConfig,
    WalkForwardSettings,
    prepare_datasets,
)


def _build_leaky_frame() -> pd.DataFrame:
    base = pd.date_range(
        "2021-01-01 00:00",
        periods=10,
        freq="15min",
        tz="UTC",
    )
    timestamps = list(base)
    timestamps[4] = timestamps[3] + pd.Timedelta(minutes=1)
    timestamps[5] = timestamps[4] + pd.Timedelta(minutes=1)
    for idx in range(6, len(timestamps)):
        timestamps[idx] = timestamps[idx - 1] + pd.Timedelta(minutes=15)

    base_values = np.linspace(1.0, 2.0, len(timestamps))
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "pair": "TEST",
            "feature_a": base_values,
            "feature_b": base_values * 2,
            "feature_c": base_values / 2,
            "target": base_values + 0.5,
        }
    )

    frame.loc[:, ["feature_a", "feature_b", "feature_c"]] = (
        frame.loc[:, ["feature_a", "feature_b", "feature_c"]]
        .shift(-1)
        .ffill()
    )
    return frame


def test_walkforward_leak_guard_blocks_overlapping_windows(tmp_path: Path) -> None:
    """Synthetic data with shifted features should trip the leak guard."""

    frame = _build_leaky_frame()
    csv_path = tmp_path / "leaky.csv"
    frame.to_csv(csv_path, index=False)

    cfg = DataConfig(
        csv_path=csv_path,
        feature_columns=["feature_a", "feature_b", "feature_c"],
        target_column="target",
        timestamp_column="timestamp",
        pair_column="pair",
        pairs=["TEST"],
        horizons=["30min"],
        time_steps=3,
        batch_size=4,
        num_workers=0,
        shuffle_train=False,
        timezone=TimezoneConfig(source="UTC", normalise_to="UTC"),
        walkforward=WalkForwardSettings(train=6, val=2, test=2, step=2, embargo=0),
    )

    with pytest.raises(ValueError, match="prediction horizon"):
        prepare_datasets(cfg)


def test_walkforward_leak_guard_allows_clean_windows(tmp_path: Path) -> None:
    """When timestamps respect the horizon, the guard should not raise."""

    frame = _build_leaky_frame()
    # Repair the timestamps so that each step is exactly 15 minutes apart.
    frame["timestamp"] = pd.date_range(
        "2021-01-01 00:00",
        periods=len(frame),
        freq="15min",
        tz="UTC",
    )
    csv_path = tmp_path / "clean.csv"
    frame.to_csv(csv_path, index=False)

    cfg = DataConfig(
        csv_path=csv_path,
        feature_columns=["feature_a", "feature_b", "feature_c"],
        target_column="target",
        timestamp_column="timestamp",
        pair_column="pair",
        pairs=["TEST"],
        horizons=["30min"],
        time_steps=3,
        batch_size=4,
        num_workers=0,
        shuffle_train=False,
        timezone=TimezoneConfig(source="UTC", normalise_to="UTC"),
        walkforward=WalkForwardSettings(train=6, val=2, test=2, step=2, embargo=0),
    )

    datasets = prepare_datasets(cfg)
    assert datasets
