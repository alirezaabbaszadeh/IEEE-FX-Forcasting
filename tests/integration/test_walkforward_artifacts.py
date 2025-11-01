from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.cli import _build_data_config, _single_run
from src.data.dataset import prepare_datasets


def _synthetic_frame(rows: int) -> pd.DataFrame:
    timestamps = pd.date_range("2021-01-01", periods=rows, freq="15min", tz="UTC")
    data = {
        "timestamp": timestamps,
        "pair": ["EURUSD"] * rows,
        "Open": [1.0 + 0.001 * i for i in range(rows)],
        "High": [1.1 + 0.001 * i for i in range(rows)],
        "Low": [0.9 + 0.001 * i for i in range(rows)],
        "Close": [1.05 + 0.001 * i for i in range(rows)],
    }
    return pd.DataFrame(data)


@pytest.mark.slow
def test_single_run_materialises_all_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_path = tmp_path / "synthetic.csv"
    frame = _synthetic_frame(rows=96)
    frame.to_csv(dataset_path, index=False)

    cfg = OmegaConf.load(Path("configs/default.yaml"))
    cfg.seed = 7
    cfg.logging.level = "WARNING"
    cfg.model.name = "TinyNet"
    cfg.data.csv_path = dataset_path.name
    cfg.data.pairs = ["EURUSD"]
    cfg.data.horizons = [1]
    cfg.data.time_steps = 8
    cfg.data.batch_size = 16
    cfg.data.num_workers = 0
    cfg.data.walkforward.train = 24
    cfg.data.walkforward.val = 8
    cfg.data.walkforward.test = 8
    cfg.data.walkforward.step = 8
    cfg.data.walkforward.embargo = 2
    cfg.training.epochs = 1
    cfg.training.device = "cpu"
    cfg.training.log_interval = 0

    monkeypatch.setattr("src.cli.get_original_cwd", lambda: str(tmp_path))

    data_cfg = _build_data_config(cfg.data, tmp_path)
    datasets = prepare_datasets(data_cfg)
    expected_window_ids = sorted(key[2] for key in datasets.keys())
    assert len(expected_window_ids) > 1

    _single_run(cfg)

    model_slug = "tinynet"
    base_dir = tmp_path / "artifacts" / "runs" / model_slug / "eurusd_1"
    assert base_dir.exists()

    for window_id in expected_window_ids:
        window_dir = base_dir / f"window-{window_id:03d}"
        assert window_dir.exists()

        metrics_path = window_dir / "metrics.json"
        metadata_path = window_dir / "metadata.json"
        assert metrics_path.exists()
        assert metadata_path.exists()

        metrics = json.loads(metrics_path.read_text())
        assert set(metrics).issuperset({"best_val_loss", "final_val_loss", "final_val_mae"})

        metadata = json.loads(metadata_path.read_text())
        dataset_meta = metadata["dataset"]
        assert "train_index" in dataset_meta
        assert "val_index" in dataset_meta
        assert "test_index" in dataset_meta
        assert dataset_meta["embargo"] == cfg.data.walkforward.embargo
        assert "calendar" in dataset_meta
        assert "embargo_gap_train_val" in dataset_meta
        assert "embargo_gap_val_test" in dataset_meta
        assert "overlaps_previous_window" in dataset_meta
