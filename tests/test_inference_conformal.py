from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.inference import PurgedConformalCalibrator, PurgedConformalConfig


def _embargo_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2023-01-01", periods=6, freq="1h", tz="UTC")
    records: list[dict[str, object]] = []
    residuals = [8.0, 7.0, 6.0, 5.0]
    for idx, residual in enumerate(residuals):
        records.append(
            {
                "pair": "EURUSD",
                "horizon": "1h",
                "window_id": 0,
                "split": "val",
                "timestamp": timestamps[idx],
                "y_true": 0.0,
                "y_pred": residual,
            }
        )
    for idx in range(4, 6):
        records.append(
            {
                "pair": "EURUSD",
                "horizon": "1h",
                "window_id": 0,
                "split": "test",
                "timestamp": timestamps[idx],
                "y_true": 0.0,
                "y_pred": 0.0,
            }
        )
    return pd.DataFrame.from_records(records)


def test_purged_conformal_respects_embargo_and_minimum_size() -> None:
    frame = _embargo_frame()
    cfg = PurgedConformalConfig(
        alpha=0.1,
        embargo=2,
        calibration_splits=("val",),
        include_past_windows=False,
        weight_decay=None,
        min_calibration=3,
    )
    calibrator = PurgedConformalCalibrator(cfg)

    intervals = calibrator.calibrate(frame)
    assert not intervals.empty

    calibration_size = intervals["calibration_size"].iloc[0]
    assert calibration_size == 3

    expected_residuals = np.array([8.0, 7.0, 6.0])
    order = np.sort(expected_residuals)
    rank = min(len(order) - 1, int(np.ceil((len(order) + 1) * (1.0 - cfg.alpha))) - 1)
    expected_radius = order[rank]
    assert intervals["calibration_radius"].iloc[0] == pytest.approx(expected_radius)


def test_recency_weights_reduce_radius() -> None:
    timestamps = pd.date_range("2023-02-01", periods=6, freq="1h", tz="UTC")
    records: list[dict[str, object]] = []
    residuals = [10.0, 9.0, 8.0, 0.5]
    for idx, residual in enumerate(residuals):
        records.append(
            {
                "pair": "EURUSD",
                "horizon": "1h",
                "window_id": 0,
                "split": "val",
                "timestamp": timestamps[idx],
                "y_true": 0.0,
                "y_pred": residual,
            }
        )
    for idx in range(4, 6):
        records.append(
            {
                "pair": "EURUSD",
                "horizon": "1h",
                "window_id": 0,
                "split": "test",
                "timestamp": timestamps[idx],
                "y_true": 0.0,
                "y_pred": 0.0,
            }
        )
    frame = pd.DataFrame.from_records(records)

    base_cfg = PurgedConformalConfig(
        alpha=0.2,
        embargo=0,
        calibration_splits=("val",),
        include_past_windows=False,
        weight_decay=None,
        min_calibration=4,
    )
    weighted_cfg = PurgedConformalConfig(
        alpha=0.2,
        embargo=0,
        calibration_splits=("val",),
        include_past_windows=False,
        weight_decay=0.01,
        min_calibration=4,
    )

    base_radius = PurgedConformalCalibrator(base_cfg).calibrate(frame)["calibration_radius"].iloc[0]
    weighted_radius = PurgedConformalCalibrator(weighted_cfg).calibrate(frame)[
        "calibration_radius"
    ].iloc[0]

    assert weighted_radius < base_radius
