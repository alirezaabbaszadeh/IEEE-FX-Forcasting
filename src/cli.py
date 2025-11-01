"""Command line interface backed by Hydra/OmegaConf."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import DataConfig, create_dataloaders, prepare_datasets
from src.experiments.multirun import RunResult, run_multirun
from src.models.forecasting import ModelConfig, TemporalForecastingModel
from src.training.engine import TrainerConfig, train
from src.utils.repro import (
    get_deterministic_flags,
    get_git_revision,
    get_hardware_snapshot,
    hash_config,
    seed_everything,
)

LOGGER = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    return str(value).replace(" ", "_").replace("/", "_").lower()


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


def _prepare_metadata(cfg: DictConfig) -> dict[str, object]:
    return {
        "config_hash": hash_config(cfg),
        "git_sha": get_git_revision(),
        "hardware": get_hardware_snapshot(),
    }


def _run_training_once(cfg: DictConfig, original_cwd: Path, metadata: dict[str, object]) -> RunResult:
    seed = int(cfg.seed)
    seed_everything(seed)
    metadata = dict(metadata)
    metadata["deterministic_flags"] = get_deterministic_flags()

    data_cfg = _build_data_config(cfg.data, original_cwd)
    LOGGER.info("Configured pairs: %s | horizons: %s", data_cfg.pairs, data_cfg.horizons)
    datasets = prepare_datasets(data_cfg)
    dataloaders = create_dataloaders(datasets, data_cfg)

    model_cfg = _build_model_config(cfg.model, len(data_cfg.feature_columns), data_cfg.time_steps)
    model = TemporalForecastingModel(model_cfg)

    trainer_cfg = _build_trainer_config(cfg.training)
    summary = train(model, dataloaders, trainer_cfg, metadata=metadata)

    return RunResult(seed=seed, summary=summary, metadata=metadata)


def _single_run(cfg: DictConfig) -> None:
    """Entry-point orchestrating a single training run."""

    original_cwd = Path(get_original_cwd())
    logging_level = getattr(logging, str(cfg.logging.level).upper(), logging.INFO)
    logging.getLogger().setLevel(logging_level)
    LOGGER.info("Loaded configuration:\n%s", OmegaConf.to_yaml(cfg))

    metadata = _prepare_metadata(cfg)
    result = _run_training_once(cfg, original_cwd, metadata)

    LOGGER.info("Best validation loss: %.4f", result.summary.best_val_loss)
    for epoch_idx, metrics in enumerate(result.summary.epochs, start=1):
        LOGGER.info(
            "Epoch %d metrics - train_loss: %.4f, val_loss: %.4f, val_mae: %.4f",
            epoch_idx,
            metrics.train_loss,
            metrics.val_loss,
            metrics.val_mae,
        )


def _resolve_output_directory(cfg: DictConfig, original_cwd: Path) -> Path:
    model_name = _slugify(cfg.model.get("name", TemporalForecastingModel.__name__))
    pair = _slugify(cfg.data.pairs[0]) if cfg.data.pairs else "all"
    horizon = _slugify(cfg.data.horizons[0]) if cfg.data.horizons else "all"
    return original_cwd / "artifacts" / "runs" / model_name / f"{pair}_{horizon}"


def _multirun_entry(cfg: DictConfig) -> None:
    original_cwd = Path(get_original_cwd())
    logging_level = getattr(logging, str(cfg.logging.level).upper(), logging.INFO)
    logging.getLogger().setLevel(logging_level)
    LOGGER.info("Loaded configuration for multirun:\n%s", OmegaConf.to_yaml(cfg))

    multirun_cfg = cfg.get("multirun")
    if multirun_cfg is None:
        seeds = [cfg.seed]
    else:
        seeds = list(multirun_cfg.get("seeds", [cfg.seed]))
    if not seeds:
        seeds = [cfg.seed]
    seeds = [int(seed) for seed in seeds]

    base_metadata = _prepare_metadata(cfg)

    output_dir = _resolve_output_directory(cfg, original_cwd)

    def _runner(seed: int) -> RunResult:
        cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_copy.seed = seed
        return _run_training_once(cfg_copy, original_cwd, base_metadata)

    aggregated = run_multirun(seeds, output_dir, _runner, base_metadata=base_metadata)
    for metric, stats in aggregated.items():
        LOGGER.info(
            "Aggregated %s - mean: %.4f | std: %.4f | ci95: %.4f",
            metric,
            stats["mean"],
            stats["std"],
            stats["ci95"],
        )


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def _hydra_single(cfg: DictConfig) -> None:
    _single_run(cfg)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def _hydra_multirun(cfg: DictConfig) -> None:
    _multirun_entry(cfg)


def main() -> None:  # pragma: no cover - thin argparse wrapper
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--multirun", action="store_true", help="launch repeated runs across seeds")
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    if args.multirun:
        _hydra_multirun()
    else:
        _hydra_single()


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    main()
