"""Command line interface backed by Hydra/OmegaConf."""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Tuple

import hydra
import pandas as pd
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from src.analysis.interpretability import MarketEvent, analyse_market_events
from src.data.dataset import (
    CalendarConfig,
    DataConfig,
    TimezoneConfig,
    WalkForwardSettings,
    WindowedData,
    create_dataloaders,
    prepare_datasets,
)
from src.experiments.multirun import RunResult, run_multirun
from src.models.forecasting import ModelConfig, TemporalForecastingModel
from src.training.engine import TrainerConfig, TrainingSummary, train
from src.utils.repro import (
    build_run_provenance,
    get_git_revision,
    get_hardware_snapshot,
    hash_config,
    seed_everything,
)

LOGGER = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    text = str(value).lower()
    result_chars: list[str] = []
    previous_was_sep = False
    for char in text:
        if char.isalnum():
            result_chars.append(char)
            previous_was_sep = False
        else:
            if not previous_was_sep:
                result_chars.append("_")
                previous_was_sep = True
    slug = "".join(result_chars).strip("_")
    return slug or "default"


def _build_data_config(cfg: DictConfig, root: Path) -> DataConfig:
    timezone_cfg = TimezoneConfig(
        source=str(cfg.timezone.source),
        normalise_to=str(cfg.timezone.normalise_to),
    )
    calendar_cfg = CalendarConfig(
        primary=cfg.calendar.get("primary"),
        fallback=cfg.calendar.get("fallback"),
    )
    walkforward_cfg = WalkForwardSettings(
        train=int(cfg.walkforward.train),
        val=int(cfg.walkforward.val),
        test=int(cfg.walkforward.test),
        step=None if cfg.walkforward.get("step") in (None, "null") else int(cfg.walkforward.step),
        embargo=int(cfg.walkforward.embargo),
    )
    return DataConfig(
        csv_path=root / cfg.csv_path,
        feature_columns=list(cfg.feature_columns),
        target_column=cfg.target_column,
        timestamp_column=cfg.timestamp_column,
        pair_column=cfg.pair_column,
        pairs=list(cfg.pairs),
        horizons=list(cfg.horizons),
        time_steps=cfg.time_steps,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle_train=cfg.shuffle_train,
        timezone=timezone_cfg,
        calendar=calendar_cfg,
        walkforward=walkforward_cfg,
    )


def _serialise_dataset_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in metadata.items():
        if isinstance(value, pd.DatetimeIndex):
            payload[key] = [ts.isoformat() for ts in value]
        elif isinstance(value, pd.Timedelta):
            payload[key] = str(value)
        elif isinstance(value, (list, tuple)):
            payload[key] = list(value)
        else:
            payload[key] = value
    return payload


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


def _summarise_training(summary: TrainingSummary) -> dict[str, float]:
    epochs = list(summary.epochs)
    final_epoch = epochs[-1] if epochs else None
    return {
        "best_val_loss": summary.best_val_loss,
        "final_val_loss": final_epoch.val_loss if final_epoch else float("nan"),
        "final_val_mae": final_epoch.val_mae if final_epoch else float("nan"),
    }


def _load_events(path: Path) -> List[MarketEvent]:
    payload = torch.load(path)
    if isinstance(payload, dict) and "events" in payload:
        raw_events = payload["events"]
    else:
        raw_events = payload

    events: List[MarketEvent] = []
    for idx, item in enumerate(raw_events):
        if isinstance(item, MarketEvent):
            events.append(item)
            continue
        if not isinstance(item, Mapping):
            raise TypeError("Events must be dictionaries or MarketEvent instances")
        if "inputs" not in item:
            raise KeyError("Each event must include an 'inputs' tensor")
        event_id = str(item.get("event_id", f"event-{idx}"))
        metadata = item.get("metadata") or {}
        events.append(
            MarketEvent(
                event_id=event_id,
                inputs=item["inputs"],
                baseline=item.get("baseline"),
                metadata=metadata,
                token_labels=item.get("token_labels"),
                feature_names=item.get("feature_names"),
            )
        )
    return events


def run_interpret_command(
    *,
    model_module: str,
    model_factory: str,
    events_path: Path,
    output_dir: Path,
    seed: int,
    limit: Optional[int] = None,
    device: Optional[str] = None,
) -> Path:
    LOGGER.info("Loading interpretability model via %s.%s", model_module, model_factory)
    seed_everything(seed)

    module = importlib.import_module(model_module)
    factory = getattr(module, model_factory, None)
    if factory is None:
        raise AttributeError(f"Factory '{model_factory}' not found in module '{model_module}'")
    model = factory()
    if not hasattr(model, "register_attention_hook"):
        raise AttributeError("Model must expose 'register_attention_hook' for interpretability")
    if not hasattr(model, "expert_activation_summaries"):
        raise AttributeError("Model must expose 'expert_activation_summaries' for interpretability")

    events = _load_events(events_path)
    if limit is not None:
        events = events[:limit]
    if not events:
        raise ValueError("No events provided for interpretability analysis")

    device_obj = torch.device(device) if device else None
    results, metadata = analyse_market_events(model, events, output_dir, device=device_obj)

    metadata = metadata.assign(
        seed=seed,
        model_module=model_module,
        model_factory=model_factory,
        event_count=len(results),
    )
    metadata_path = Path(output_dir) / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    LOGGER.info("Saved interpretability metadata to %s", metadata_path)
    return metadata_path


def _run_training_once(
    cfg: DictConfig,
    original_cwd: Path,
    metadata: dict[str, object],
    *,
    dataset_key: Tuple[str, object, int] | None = None,
    dataset: WindowedData | None = None,
    data_cfg: DataConfig | None = None,
) -> RunResult:
    seed = int(cfg.seed)
    seed_everything(seed)
    data_cfg_obj = data_cfg or _build_data_config(cfg.data, original_cwd)
    LOGGER.info("Configured pairs: %s | horizons: %s", data_cfg_obj.pairs, data_cfg_obj.horizons)

    selected_key: Tuple[str, object, int]
    selected_window: WindowedData
    if dataset_key is not None and dataset is not None:
        selected_key = dataset_key
        selected_window = dataset
    else:
        datasets = prepare_datasets(data_cfg_obj)
        selected_key, selected_window = next(iter(datasets.items()))
        LOGGER.info(
            "Using window %s for initial training loop (total windows: %d)",
            selected_key,
            len(datasets),
        )

    dataloaders = create_dataloaders(selected_window, data_cfg_obj)

    run_metadata = build_run_provenance(
        seed,
        metadata,
        dataset={
            "key": {
                "pair": selected_key[0],
                "horizon": str(selected_key[1]),
                "window_id": selected_key[2],
            },
            **_serialise_dataset_metadata(selected_window.metadata),
        },
    )

    model_cfg = _build_model_config(
        cfg.model,
        len(data_cfg_obj.feature_columns),
        data_cfg_obj.time_steps,
    )
    model = TemporalForecastingModel(model_cfg)

    trainer_cfg = _build_trainer_config(cfg.training)
    summary = train(model, dataloaders, trainer_cfg, metadata=dict(run_metadata))

    return RunResult(seed=seed, summary=summary, metadata=dict(run_metadata))


def _single_run(cfg: DictConfig) -> None:
    """Entry-point orchestrating a single training run."""

    original_cwd = Path(get_original_cwd())
    logging_level = getattr(logging, str(cfg.logging.level).upper(), logging.INFO)
    logging.getLogger().setLevel(logging_level)
    LOGGER.info("Loaded configuration:\n%s", OmegaConf.to_yaml(cfg))

    metadata = _prepare_metadata(cfg)
    data_cfg = _build_data_config(cfg.data, original_cwd)
    datasets = prepare_datasets(data_cfg)

    for dataset_key, window in datasets.items():
        dataset_metadata = dict(window.metadata)
        dataset_metadata.setdefault("pair", dataset_key[0])
        dataset_metadata.setdefault("horizon", dataset_key[1])
        dataset_metadata.setdefault("window_id", dataset_key[2])

        LOGGER.info(
            "Starting training for pair=%s horizon=%s window=%s",
            dataset_key[0],
            dataset_key[1],
            dataset_key[2],
        )

        output_dir = _resolve_output_directory(cfg, original_cwd, dataset=dataset_metadata)
        result = _run_training_once(
            cfg,
            original_cwd,
            metadata,
            dataset_key=dataset_key,
            dataset=window,
            data_cfg=data_cfg,
        )
        _write_single_run_artifacts(output_dir, result)

        LOGGER.info("Best validation loss: %.4f", result.summary.best_val_loss)
        for epoch_idx, metrics in enumerate(result.summary.epochs, start=1):
            LOGGER.info(
                "Epoch %d metrics - train_loss: %.4f, val_loss: %.4f, val_mae: %.4f",
                epoch_idx,
                metrics.train_loss,
                metrics.val_loss,
                metrics.val_mae,
            )


def _resolve_output_directory(
    cfg: DictConfig,
    original_cwd: Path,
    *,
    dataset: Mapping[str, object] | None = None,
) -> Path:
    model_name = _slugify(cfg.model.get("name", TemporalForecastingModel.__name__))
    base = original_cwd / "artifacts" / "runs" / model_name
    if dataset is None:
        pair = _slugify(cfg.data.pairs[0]) if cfg.data.pairs else "all"
        horizon = _slugify(cfg.data.horizons[0]) if cfg.data.horizons else "all"
        combined = f"{pair}_{horizon}"
        return base / combined / "window-000"

    pair_slug = _slugify(dataset.get("pair", "all"))
    horizon_value = dataset.get("horizon_steps") or dataset.get("horizon") or "all"
    horizon_slug = _slugify(str(horizon_value))
    combined = f"{pair_slug}_{horizon_slug}"
    window_id = dataset.get("window_id")
    window_slug = f"window-{int(window_id):03d}" if window_id is not None else "window-000"
    return base / combined / window_slug


def _write_single_run_artifacts(output_dir: Path, result: RunResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _summarise_training(result.summary)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    metadata = dict(result.metadata)
    metadata["seed"] = result.seed
    metadata.setdefault("device", result.summary.device)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    LOGGER.info("Persisted training artifacts to %s", output_dir)


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

    data_cfg = _build_data_config(cfg.data, original_cwd)
    datasets = prepare_datasets(data_cfg)

    for dataset_key, window in datasets.items():
        dataset_metadata = dict(window.metadata)
        dataset_metadata.setdefault("pair", dataset_key[0])
        dataset_metadata.setdefault("horizon", dataset_key[1])
        dataset_metadata.setdefault("window_id", dataset_key[2])

        output_dir = _resolve_output_directory(cfg, original_cwd, dataset=dataset_metadata)

        def _runner(seed: int, *, _dataset_key=dataset_key, _window=window) -> RunResult:
            cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            cfg_copy.seed = seed
            return _run_training_once(
                cfg_copy,
                original_cwd,
                base_metadata,
                dataset_key=_dataset_key,
                dataset=_window,
                data_cfg=data_cfg,
            )

        LOGGER.info(
            "Launching multirun for pair=%s horizon=%s window=%s", *dataset_key
        )
        dataset_base_metadata = dict(base_metadata)
        dataset_base_metadata["dataset"] = {
            "pair": dataset_key[0],
            "horizon": str(dataset_key[1]),
            "window_id": dataset_key[2],
        }
        aggregated = run_multirun(
            seeds,
            output_dir,
            _runner,
            base_metadata=dataset_base_metadata,
        )
        for metric, stats in aggregated.items():
            LOGGER.info(
                "Aggregated %s - mean: %.4f | std: %.4f | ci95: %.4f",  # type: ignore[index]
                metric,
                stats.get("mean", float("nan")),
                stats.get("std", float("nan")),
                stats.get("ci95", float("nan")),
            )


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def _hydra_single(cfg: DictConfig) -> None:
    _single_run(cfg)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def _hydra_multirun(cfg: DictConfig) -> None:
    _multirun_entry(cfg)


def main() -> None:  # pragma: no cover - thin argparse wrapper
    if len(sys.argv) > 1 and sys.argv[1] == "interpret":
        parser = argparse.ArgumentParser(description="Run interpretability analyses")
        parser.add_argument("interpret", help=argparse.SUPPRESS)
        parser.add_argument("--model-module", required=True, help="Module path exposing the model factory")
        parser.add_argument(
            "--model-factory",
            default="build_model",
            help="Factory function returning an instantiated model",
        )
        parser.add_argument("--events", required=True, type=Path, help="Path to a torch-saved events file")
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("artifacts/interpretability"),
            help="Directory where artefacts will be written",
        )
        parser.add_argument("--seed", type=int, default=7, help="Seed for deterministic artefacts")
        parser.add_argument("--limit", type=int, help="Optional limit on the number of events to process")
        parser.add_argument("--device", help="Optional torch device for execution")
        args = parser.parse_args()

        run_interpret_command(
            model_module=args.model_module,
            model_factory=args.model_factory,
            events_path=args.events,
            output_dir=args.output_dir,
            seed=args.seed,
            limit=args.limit,
            device=args.device,
        )
        return

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
