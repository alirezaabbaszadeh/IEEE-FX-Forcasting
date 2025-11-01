from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.cli import (
    _build_data_config,
    _build_model_config,
    _build_trainer_config,
    _prepare_metadata,
)
from src.data.dataset import create_dataloaders, prepare_datasets
from src.models.forecasting import TemporalForecastingModel
from src.training.engine import train
from src.utils.repro import get_deterministic_flags, seed_everything


@pytest.mark.slow
def test_single_epoch_smoke_run_is_deterministic() -> None:
    cfg = OmegaConf.load(Path("configs/default.yaml"))
    cfg.training.epochs = 1
    cfg.training.device = "cpu"
    cfg.data.time_steps = 16
    cfg.data.batch_size = 32

    seed_everything(int(cfg.seed))
    metadata = _prepare_metadata(cfg)
    metadata["deterministic_flags"] = get_deterministic_flags()

    project_root = Path.cwd()
    data_cfg = _build_data_config(cfg.data, project_root)
    datasets = prepare_datasets(data_cfg)
    window = next(iter(datasets.values()))
    dataloaders = create_dataloaders(window, data_cfg)
    model_cfg = _build_model_config(cfg.model, len(data_cfg.feature_columns), data_cfg.time_steps)
    trainer_cfg = _build_trainer_config(cfg.training)

    summary = train(TemporalForecastingModel(model_cfg), dataloaders, trainer_cfg, metadata)

    assert summary.best_val_loss == pytest.approx(0.20910307268301645, rel=1e-2)
    epoch_metrics = summary.epochs[-1]
    assert epoch_metrics.train_loss == pytest.approx(0.3642687678337097, rel=1e-2)
    assert epoch_metrics.val_loss == pytest.approx(0.20910307268301645, rel=1e-2)
    assert epoch_metrics.val_mae == pytest.approx(0.366701861222585, rel=1e-2)
