from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.eval.run import run_evaluation
from src.inference import PurgedConformalConfig
from src.metrics.calibration import interval_coverage


def _synthetic_predictions(
    *,
    windows: int = 3,
    val_size: int = 60,
    test_size: int = 40,
    embargo: int = 2,
) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    base_time = pd.Timestamp("2023-04-01", tz="UTC")
    step = pd.Timedelta(minutes=15)
    offset = 0
    records: list[dict[str, object]] = []
    for window_id in range(windows):
        for idx in range(val_size):
            ts = base_time + (offset + idx) * step
            y_true = rng.normal()
            y_pred = y_true + rng.normal(scale=0.5)
            records.append(
                {
                    "pair": "EURUSD",
                    "horizon": "1h",
                    "window_id": window_id,
                    "split": "val",
                    "timestamp": ts,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
        offset += val_size + embargo
        for idx in range(test_size):
            ts = base_time + (offset + idx) * step
            y_true = rng.normal()
            y_pred = y_true + rng.normal(scale=0.5)
            records.append(
                {
                    "pair": "EURUSD",
                    "horizon": "1h",
                    "window_id": window_id,
                    "split": "test",
                    "timestamp": ts,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
        offset += test_size
    return pd.DataFrame.from_records(records)


def test_run_evaluation_produces_intervals_with_expected_coverage(tmp_path: Path) -> None:
    frame = _synthetic_predictions()
    predictions_path = tmp_path / "predictions.csv"
    frame.to_csv(predictions_path, index=False)

    freeze_manifest = tmp_path / "claim_freeze.yaml"
    freeze_manifest.write_text("frozen_at: 2023-03-01T00:00:00Z\nnotes: synthetic validation\n")

    raw_cfg = OmegaConf.load(Path("configs/inference/pcc.yaml"))
    cfg_mapping = OmegaConf.to_container(raw_cfg, resolve=True)
    assert isinstance(cfg_mapping, dict)
    calibration_cfg = PurgedConformalConfig.from_mapping(cfg_mapping)

    metrics_path = run_evaluation(
        predictions_path=predictions_path,
        run_id="synthetic",
        artifacts_dir=tmp_path,
        calibration_cfg=calibration_cfg,
        claim_freeze_manifest=freeze_manifest,
    )

    assert metrics_path.exists()
    intervals_path = tmp_path / "synthetic" / "intervals.csv"
    assert intervals_path.exists()

    freeze_output = tmp_path / "synthetic" / "claim_freeze.json"
    assert freeze_output.exists()

    intervals = pd.read_csv(intervals_path)
    coverage = interval_coverage(
        intervals["y_true"], intervals["interval_lower"], intervals["interval_upper"]
    )
    nominal = 1.0 - calibration_cfg.alpha
    assert coverage == pytest.approx(nominal, abs=0.05)

    freq = frame.sort_values("timestamp")["timestamp"].diff().dropna().min()
    assert isinstance(freq, pd.Timedelta)
    for window_id, window_rows in intervals.groupby("window_id"):
        if window_rows.empty:
            continue
        min_test_ts = pd.to_datetime(window_rows["timestamp"], utc=True).min()
        last_cal_ts = pd.to_datetime(window_rows["calibration_last_timestamp"], utc=True).max()
        if pd.isna(last_cal_ts):
            continue
        assert min_test_ts - last_cal_ts >= freq * calibration_cfg.embargo


def test_run_evaluation_rejects_pre_freeze_test_access(tmp_path: Path) -> None:
    frame = _synthetic_predictions()
    predictions_path = tmp_path / "predictions.csv"
    frame.to_csv(predictions_path, index=False)

    manifest = tmp_path / "claim_freeze.yaml"
    manifest.write_text("frozen_at: 2025-01-01T00:00:00Z\n")

    with pytest.raises(ValueError, match="precede the claim freeze"):
        run_evaluation(
            predictions_path=predictions_path,
            run_id="freeze_failure",
            artifacts_dir=tmp_path,
            claim_freeze_manifest=manifest,
        )
