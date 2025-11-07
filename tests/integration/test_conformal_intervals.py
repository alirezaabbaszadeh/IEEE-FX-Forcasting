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


def test_run_evaluation_projects_crossing_intervals(monkeypatch, tmp_path: Path) -> None:
    timestamps = pd.date_range("2024-01-01", periods=3, freq="1H", tz="UTC")
    frame = pd.DataFrame(
        {
            "pair": ["EURUSD"] * 3,
            "horizon": ["1h"] * 3,
            "timestamp": timestamps,
            "y_true": [0.0, 0.1, -0.2],
            "y_pred": [0.05, -0.05, 0.02],
            "split": ["test", "test", "test"],
            "quantile_0.10": [0.2, -0.4, 0.3],
            "quantile_0.90": [0.1, -0.8, 0.1],
        }
    )
    predictions_path = tmp_path / "rcqf_predictions.csv"
    frame.to_csv(predictions_path, index=False)

    captured: dict[str, pd.DataFrame] = {}

    def _fake_calibrate(self, predictions: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        captured["predictions"] = predictions.copy()
        base = predictions.iloc[[0]].copy()
        base["interval_lower"] = 1.0
        base["interval_upper"] = 0.0
        base["calibration_radius"] = 0.0
        base["calibration_size"] = 1
        base["calibration_weight_sum"] = 1.0
        base["calibration_last_timestamp"] = base["timestamp"]
        base["alpha"] = 0.1
        return base

    monkeypatch.setattr(
        "src.inference.conformal_purged.PurgedConformalCalibrator.calibrate",
        _fake_calibrate,
    )

    cfg = PurgedConformalConfig(alpha=0.1, calibration_splits=("test",), include_past_windows=False)

    run_evaluation(
        predictions_path=predictions_path,
        run_id="quantile_fix",
        artifacts_dir=tmp_path,
        calibration_cfg=cfg,
    )

    predictions_seen = captured.get("predictions")
    assert predictions_seen is not None
    quantile_values = predictions_seen[["quantile_0.10", "quantile_0.90"]].to_numpy(dtype=float)
    assert np.all(np.diff(quantile_values, axis=1) >= -1e-12)

    intervals_path = tmp_path / "quantile_fix" / "intervals.csv"
    intervals = pd.read_csv(intervals_path)
    assert np.all(intervals["interval_lower"] <= intervals["interval_upper"] + 1e-12)
    assert intervals["interval_lower"].iloc[0] == pytest.approx(intervals["interval_upper"].iloc[0])
