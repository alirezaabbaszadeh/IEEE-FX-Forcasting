"""Command line interface backed by Hydra/OmegaConf."""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import DataConfig, create_dataloaders, prepare_datasets
from src.models.forecasting import ModelConfig, TemporalForecastingModel
from src.training.engine import TrainerConfig, train
from src.utils.repro import seed_everything

LOGGER = logging.getLogger(__name__)


def _build_data_config(cfg: DictConfig, root: Path) -> DataConfig:
    return DataConfig(
        csv_path=root / cfg.csv_path,
        feature_columns=list(cfg.feature_columns),
        target_column=cfg.target_column,
        pairs=list(cfg.pairs),
        horizons=list(cfg.horizons),
        time_steps=cfg.time_steps,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle_train=cfg.shuffle_train,
    )


def _build_model_config(cfg: DictConfig, input_features: int, time_steps: int) -> ModelConfig:
    return ModelConfig(
        input_features=input_features,
        time_steps=time_steps,
        hidden_size=cfg.hidden_size,
        conv_kernel_size=cfg.conv_kernel_size,
        num_blocks=cfg.num_blocks,
        attention_heads=cfg.attention_heads,
        lstm_hidden_size=cfg.lstm_hidden_size,
        dropout=cfg.dropout,
    )


def _build_trainer_config(cfg: DictConfig) -> TrainerConfig:
    return TrainerConfig(
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
        log_interval=cfg.log_interval,
        device=cfg.device,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Entry-point orchestrating data preparation, model instantiation and training."""

    original_cwd = Path(get_original_cwd())
    logging_level = getattr(logging, str(cfg.logging.level).upper(), logging.INFO)
    logging.getLogger().setLevel(logging_level)
    LOGGER.info("Loaded configuration:\n%s", OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed)

    data_cfg = _build_data_config(cfg.data, original_cwd)
    LOGGER.info("Configured pairs: %s | horizons: %s", data_cfg.pairs, data_cfg.horizons)
    datasets = prepare_datasets(data_cfg)
    dataloaders = create_dataloaders(datasets, data_cfg)

    model_cfg = _build_model_config(cfg.model, len(data_cfg.feature_columns), data_cfg.time_steps)
    model = TemporalForecastingModel(model_cfg)

    trainer_cfg = _build_trainer_config(cfg.training)
    summary = train(model, dataloaders, trainer_cfg)

    LOGGER.info("Best validation loss: %.4f", summary.best_val_loss)
    for epoch_idx, metrics in enumerate(summary.epochs, start=1):
        LOGGER.info(
            "Epoch %d metrics - train_loss: %.4f, val_loss: %.4f, val_mae: %.4f",
            epoch_idx,
            metrics.train_loss,
            metrics.val_loss,
            metrics.val_mae,
        )


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    main()
