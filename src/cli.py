"""Command line interface backed by Hydra/OmegaConf."""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Tuple

import hydra
import pandas as pd
import torch
from torch import nn
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
from src.utils.artifacts import (
    build_run_metadata,
    collect_environment_lockfiles,
    compute_dataset_checksums,
    ensure_config_snapshot,
)
from src.utils.repro import (
    build_run_provenance,
    get_git_revision,
    get_hardware_snapshot,
    seed_everything,
)

LOGGER = logging.getLogger(__name__)


_BENCHMARK_MODE: str | None = None
_BENCHMARK_PRESETS: dict[str, dict[str, int]] = {
    "smoke": {
        "train_warmup": 1,
        "train_steps": 3,
        "inference_warmup": 1,
        "inference_steps": 5,
    },
    "full": {
        "train_warmup": 5,
        "train_steps": 20,
        "inference_warmup": 5,
        "inference_steps": 50,
    },
}


def _set_benchmark_mode(mode: str | None) -> None:
    global _BENCHMARK_MODE
    _BENCHMARK_MODE = mode


def _get_benchmark_mode() -> str | None:
    return _BENCHMARK_MODE


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


def _persist_split_audit(output_dir: Path, dataset_metadata: Mapping[str, object]) -> None:
    """Write a CSV summary of the dataset splits if diagnostics are available."""

    records = dataset_metadata.get("split_records")
    if not records:
        return

    frame = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "splits.csv"
    frame.to_csv(path, index=False)
    LOGGER.info("Saved split audit to %s", path)


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


def _build_light_lstm_config(cfg: DictConfig, input_features: int, time_steps: int):
    from src.models.deep.light_lstm import LightLSTMConfig

    return LightLSTMConfig(
        input_features=input_features,
        time_steps=time_steps,
        hidden_size=int(cfg.get("hidden_size", 64)),
        num_layers=int(cfg.get("num_layers", 1)),
        dropout=float(cfg.get("dropout", 0.1)),
        bidirectional=bool(cfg.get("bidirectional", False)),
    )


def _to_optional_tuple(values: object, expected_len: int) -> tuple[int, ...] | None:
    if values in (None, "null"):
        return None
    iterable = list(values)
    if len(iterable) != expected_len:
        raise ValueError(f"Expected sequence of length {expected_len}, received {iterable}")
    return tuple(int(v) for v in iterable)


def _normalise_optional_str(value: object | None) -> str | None:
    if value in (None, "null"):
        return None
    text = str(value)
    return text if text else None


def _build_arima_config(cfg: DictConfig):
    from src.models.classical import ArimaConfig

    order = tuple(int(v) for v in list(cfg.get("order", (1, 1, 0))))
    seasonal_order = _to_optional_tuple(cfg.get("seasonal_order"), 4)
    method = _normalise_optional_str(cfg.get("method"))
    maxiter = cfg.get("maxiter")
    maxiter_val = None if maxiter in (None, "null") else int(maxiter)
    return ArimaConfig(
        order=order,
        seasonal_order=seasonal_order,
        trend=_normalise_optional_str(cfg.get("trend")),
        enforce_stationarity=bool(cfg.get("enforce_stationarity", True)),
        enforce_invertibility=bool(cfg.get("enforce_invertibility", True)),
        method=method,
        maxiter=maxiter_val,
    )


def _build_ets_config(cfg: DictConfig):
    from src.models.classical import ETSConfig

    seasonal_periods = cfg.get("seasonal_periods")
    periods_val = None if seasonal_periods in (None, "null") else int(seasonal_periods)
    use_boxcox = cfg.get("use_boxcox")
    if isinstance(use_boxcox, str) and use_boxcox.lower() == "none":
        use_boxcox_val: float | bool | None = None
    else:
        use_boxcox_val = use_boxcox  # type: ignore[assignment]

    return ETSConfig(
        trend=_normalise_optional_str(cfg.get("trend", "add")),
        damped_trend=bool(cfg.get("damped_trend", False)),
        seasonal=_normalise_optional_str(cfg.get("seasonal")),
        seasonal_periods=periods_val,
        use_boxcox=use_boxcox_val,
        initialization_method=str(cfg.get("initialization_method", "estimated")),
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


def _prepare_metadata(cfg: DictConfig, *, original_cwd: Path) -> dict[str, object]:
    artifacts_root = original_cwd / "artifacts"
    config_info = ensure_config_snapshot(cfg, artifacts_root)
    lockfiles = collect_environment_lockfiles(original_cwd)
    metadata: dict[str, object] = {
        "config": {
            "hash": config_info["hash"],
            "path": str(Path(config_info["path"]).relative_to(original_cwd)),
        },
        "config_hash": config_info["hash"],
        "git_sha": get_git_revision(),
        "hardware": get_hardware_snapshot(),
    }
    if lockfiles:
        metadata["environment"] = {"lockfiles": lockfiles}
    return metadata


def _run_post_training_benchmarks(
    model: nn.Module,
    dataloaders: Mapping[str, object],
    trainer_cfg: TrainerConfig,
    *,
    mode: str,
) -> dict[str, "BenchmarkReport"]:
    try:
        from src.analysis import BenchmarkReport, benchmark_model
    except ImportError:  # pragma: no cover - optional dependency
        LOGGER.warning("Skipping benchmarking because analysis utilities are unavailable")
        return {}

    presets = _BENCHMARK_PRESETS.get(mode)
    if presets is None:
        LOGGER.warning("Unknown benchmarking mode '%s' - skipping", mode)
        return {}

    device = next(model.parameters()).device
    reports: dict[str, BenchmarkReport] = {}

    train_loader = dataloaders.get("train")
    if train_loader is not None and presets.get("train_steps", 0) > 0:
        dataset = getattr(train_loader, "dataset", None)
        length = len(dataset) if hasattr(dataset, "__len__") else None
        if length is None or length > 0:
            LOGGER.info("Running training benchmark (%s mode)", mode)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=trainer_cfg.learning_rate,
                weight_decay=trainer_cfg.weight_decay,
            )
            loss_fn = nn.MSELoss()
            snapshot = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            report = benchmark_model(
                model,
                train_loader,
                mode="training",
                warmup_steps=presets.get("train_warmup", 0),
                measure_steps=presets.get("train_steps", 0),
                device=device,
                loss_fn=loss_fn,
                optimizer=optimizer,
                dataloader_label="train",
            )
            reports["train_training"] = report
            model.load_state_dict(snapshot)
            for parameter in model.parameters():
                parameter.grad = None

    inference_steps = presets.get("inference_steps", 0)
    if inference_steps <= 0:
        return reports

    for split in ("val", "test"):
        loader = dataloaders.get(split)
        if loader is None:
            continue
        dataset = getattr(loader, "dataset", None)
        length = len(dataset) if hasattr(dataset, "__len__") else None
        if length is not None and length == 0:
            continue
        LOGGER.info("Running inference benchmark on %s split (%s mode)", split, mode)
        report = benchmark_model(
            model,
            loader,
            mode="inference",
            warmup_steps=presets.get("inference_warmup", 0),
            measure_steps=inference_steps,
            device=device,
            dataloader_label=split,
        )
        reports[f"{split}_inference"] = report

    return reports


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
    pair: Optional[str] = None,
    horizon: Optional[object] = None,
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
    derived_pair = pair
    if derived_pair is None:
        for event in events:
            event_pair = event.metadata.get("pair") if event.metadata else None
            if event_pair is not None:
                derived_pair = str(event_pair)
                break
    derived_pair = derived_pair or "unknown"

    derived_horizon = horizon
    if derived_horizon is None:
        for event in events:
            event_metadata = event.metadata or {}
            horizon_value = (
                event_metadata.get("horizon")
                or event_metadata.get("horizon_steps")
                or event_metadata.get("horizon_label")
            )
            if horizon_value is not None:
                derived_horizon = str(horizon_value)
                break
    derived_horizon = derived_horizon or "unknown"

    structured_root = (
        Path(output_dir)
        / _slugify(derived_pair)
        / _slugify(str(derived_horizon))
        / f"seed-{seed}"
    )
    structured_root.mkdir(parents=True, exist_ok=True)

    results, metadata = analyse_market_events(model, events, structured_root, device=device_obj)

    metadata = metadata.assign(
        seed=seed,
        model_module=model_module,
        model_factory=model_factory,
        event_count=len(results),
        pair=derived_pair,
        horizon=derived_horizon,
    )
    metadata_path = Path(structured_root) / "metadata.csv"
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
    manifest_path: Path | None = None,
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

    model_name = str(cfg.model.get("name", "temporal_transformer")).lower()
    lookback = data_cfg_obj.time_steps
    raw_horizon_steps = selected_window.metadata.get("horizon_steps")
    try:
        horizon_steps = int(raw_horizon_steps)
    except (TypeError, ValueError):
        horizon_steps = 1

    model = None
    trainer_cfg = None
    baseline_metrics: dict[str, object] | None = None
    summary: TrainingSummary | None = None

    if model_name in {"temporal_transformer", "transformer", "temporalforecastingmodel"}:
        model_cfg = _build_model_config(
            cfg.model,
            len(data_cfg_obj.feature_columns),
            lookback,
        )
        model = TemporalForecastingModel(model_cfg)
    elif model_name == "lstm_light":
        from src.models.deep.light_lstm import LightLSTMModel

        lstm_cfg = _build_light_lstm_config(
            cfg.model,
            len(data_cfg_obj.feature_columns),
            lookback,
        )
        model = LightLSTMModel(lstm_cfg)
    elif model_name == "arima":
        from src.models.classical import run_arima_baseline

        arima_cfg = _build_arima_config(cfg.model)
        summary, baseline_metrics = run_arima_baseline(
            selected_window,
            lookback=lookback,
            horizon_steps=horizon_steps,
            config=arima_cfg,
        )
    elif model_name == "ets":
        from src.models.classical import run_ets_baseline

        ets_cfg = _build_ets_config(cfg.model)
        summary, baseline_metrics = run_ets_baseline(
            selected_window,
            lookback=lookback,
            horizon_steps=horizon_steps,
            config=ets_cfg,
        )
    else:
        raise ValueError(f"Unsupported model baseline: {model_name}")

    if model is not None:
        trainer_cfg = _build_trainer_config(cfg.training)
        summary = train(
            model,
            dataloaders,
            trainer_cfg,
            metadata=dict(run_metadata),
            manifest_path=manifest_path,
        )

    benchmark_mode = _get_benchmark_mode()
    if benchmark_mode and model is not None and trainer_cfg is not None:
        LOGGER.info("Benchmarking enabled (%s mode)", benchmark_mode)
        benchmark_reports = _run_post_training_benchmarks(
            model,
            dataloaders,
            trainer_cfg,
            mode=benchmark_mode,
        )
        if benchmark_reports:
            summary.benchmarks.update(benchmark_reports)

    if summary is None:
        raise RuntimeError("Training summary was not produced for model run")

    if baseline_metrics:
        LOGGER.info("Baseline metrics (scaled targets): %s", baseline_metrics)
        run_metadata.setdefault("baseline_metrics", baseline_metrics)

    return RunResult(seed=seed, summary=summary, metadata=dict(run_metadata))


def _resolve_interpret_runs(
    cfg: DictConfig,
    dataset_metadata: Mapping[str, object],
    *,
    seed: int,
) -> List[dict[str, Any]]:
    interpret_cfg = cfg.get("interpret")
    if interpret_cfg is None:
        return []

    interpret_container = OmegaConf.to_container(interpret_cfg, resolve=True)
    if not isinstance(interpret_container, Mapping):
        return []

    if not interpret_container.get("enabled", False):
        return []

    runs = interpret_container.get("runs") or []
    if not isinstance(runs, list):
        return []

    defaults = {
        "model_module": interpret_container.get("model_module"),
        "model_factory": interpret_container.get("model_factory", "build_model"),
        "output_dir": interpret_container.get("output_dir", "artifacts/interpretability"),
        "device": interpret_container.get("device"),
        "limit": interpret_container.get("limit"),
        "seed": interpret_container.get("seed"),
    }

    resolved: List[dict[str, Any]] = []
    dataset_pair = dataset_metadata.get("pair")
    dataset_horizon = dataset_metadata.get("horizon")

    for entry in runs:
        if not isinstance(entry, Mapping):
            continue

        pair_filter = entry.get("pair")
        if pair_filter is not None and str(pair_filter) != str(dataset_pair):
            continue

        horizon_filter = entry.get("horizon")
        if horizon_filter is not None and str(horizon_filter) != str(dataset_horizon):
            continue

        events_path = entry.get("events")
        if not events_path:
            LOGGER.warning(
                "Skipping interpret run without events path for pair=%s horizon=%s",
                dataset_pair,
                dataset_horizon,
            )
            continue

        model_module = entry.get("model_module", defaults["model_module"])
        if not model_module:
            LOGGER.warning("Skipping interpret run for %s due to missing model_module", events_path)
            continue

        resolved.append(
            {
                "model_module": model_module,
                "model_factory": entry.get("model_factory", defaults["model_factory"]),
                "events": events_path,
                "output_dir": entry.get("output_dir", defaults["output_dir"]),
                "limit": entry.get("limit", defaults["limit"]),
                "device": entry.get("device", defaults["device"]),
                "seed": entry.get("seed", defaults["seed"]) or seed,
            }
        )

    return resolved


def _execute_interpret_runs(
    cfg: DictConfig,
    dataset_metadata: Mapping[str, object],
    *,
    seed: int,
    original_cwd: Path,
) -> None:
    runs = _resolve_interpret_runs(cfg, dataset_metadata, seed=seed)
    if not runs:
        return

    pair_value = dataset_metadata.get("pair")
    horizon_value = dataset_metadata.get("horizon")

    for run_cfg in runs:
        events_path = Path(run_cfg["events"])
        if not events_path.is_absolute():
            events_path = original_cwd / events_path

        output_dir = Path(run_cfg["output_dir"])
        if not output_dir.is_absolute():
            output_dir = original_cwd / output_dir

        run_seed = int(run_cfg["seed"])
        LOGGER.info(
            "Running interpretability for pair=%s horizon=%s seed=%d using events %s",
            pair_value,
            horizon_value,
            run_seed,
            events_path,
        )
        run_interpret_command(
            model_module=str(run_cfg["model_module"]),
            model_factory=str(run_cfg["model_factory"]),
            events_path=events_path,
            output_dir=output_dir,
            seed=run_seed,
            limit=int(run_cfg["limit"]) if run_cfg.get("limit") is not None else None,
            device=str(run_cfg["device"]) if run_cfg.get("device") is not None else None,
            pair=str(pair_value) if pair_value is not None else None,
            horizon=str(horizon_value) if horizon_value is not None else None,
        )


def _single_run(cfg: DictConfig) -> None:
    """Entry-point orchestrating a single training run."""

    original_cwd = Path(get_original_cwd())
    logging_level = getattr(logging, str(cfg.logging.level).upper(), logging.INFO)
    logging.getLogger().setLevel(logging_level)
    LOGGER.info("Loaded configuration:\n%s", OmegaConf.to_yaml(cfg))

    metadata = _prepare_metadata(cfg, original_cwd=original_cwd)
    config_hash = str(metadata.get("config_hash"))
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

        output_dir = _resolve_output_directory(
            cfg,
            original_cwd,
            dataset=dataset_metadata,
            config_hash=config_hash,
        )
        result = _run_training_once(
            cfg,
            original_cwd,
            metadata,
            dataset_key=dataset_key,
            dataset=window,
            data_cfg=data_cfg,
            manifest_path=output_dir / "manifest.json",
        )
        _write_single_run_artifacts(output_dir, result)

        _execute_interpret_runs(
            cfg,
            dataset_metadata,
            seed=result.seed,
            original_cwd=original_cwd,
        )

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
    config_hash: str,
) -> Path:
    model_name = _slugify(cfg.model.get("name", TemporalForecastingModel.__name__))
    base = original_cwd / "artifacts" / "runs" / model_name / config_hash
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
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    metadata = dict(result.metadata)
    dataset_meta = dict(metadata.get("dataset") or {})
    _persist_split_audit(output_dir, dataset_meta)

    artifact_index: dict[str, object] = {
        "metrics": metrics_path.name,
        "metadata": "metadata.json",
        "manifest": "manifest.json",
    }

    if dataset_meta.get("split_records"):
        artifact_index["splits"] = "splits.csv"

    if getattr(result.summary, "benchmarks", None):
        try:
            from src.analysis import save_report
        except ImportError:  # pragma: no cover - optional dependency
            LOGGER.warning("Benchmarks collected but analysis utilities unavailable; skipping save")
        else:
            benchmarks_dir = output_dir / "benchmarks"
            benchmark_entries: list[dict[str, str]] = []
            for name, report in result.summary.benchmarks.items():
                saved_paths = save_report(report, benchmarks_dir, stem=name)
                benchmark_entries.append(
                    {
                        "name": name,
                        "json": str(saved_paths.json_path.relative_to(output_dir)),
                        "csv": str(saved_paths.csv_path.relative_to(output_dir)),
                    }
                )
            if benchmark_entries:
                artifact_index["benchmarks"] = benchmark_entries

    dataset_checksums = compute_dataset_checksums(dataset_meta)
    metadata_payload = build_run_metadata(
        metadata,
        seed=result.seed,
        device=result.summary.device,
        artifact_index=artifact_index,
        dataset_checksums=dataset_checksums,
    )

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_payload, handle, indent=2, sort_keys=True)

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

    base_metadata = _prepare_metadata(cfg, original_cwd=original_cwd)
    config_hash = str(base_metadata.get("config_hash"))

    data_cfg = _build_data_config(cfg.data, original_cwd)
    datasets = prepare_datasets(data_cfg)

    for dataset_key, window in datasets.items():
        dataset_metadata = dict(window.metadata)
        dataset_metadata.setdefault("pair", dataset_key[0])
        dataset_metadata.setdefault("horizon", dataset_key[1])
        dataset_metadata.setdefault("window_id", dataset_key[2])

        output_dir = _resolve_output_directory(
            cfg,
            original_cwd,
            dataset=dataset_metadata,
            config_hash=config_hash,
        )

        def _runner(
            seed: int,
            *,
            _dataset_key=dataset_key,
            _window=window,
            _output_dir=output_dir,
        ) -> RunResult:
            cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            cfg_copy.seed = seed
            result = _run_training_once(
                cfg_copy,
                original_cwd,
                base_metadata,
                dataset_key=_dataset_key,
                dataset=_window,
                data_cfg=data_cfg,
                manifest_path=(_output_dir / f"seed-{seed}" / "manifest.json"),
            )
            dataset_meta = dict(_window.metadata)
            dataset_meta.setdefault("pair", _dataset_key[0])
            dataset_meta.setdefault("horizon", _dataset_key[1])
            dataset_meta.setdefault("window_id", _dataset_key[2])
            _execute_interpret_runs(
                cfg_copy,
                dataset_meta,
                seed=result.seed,
                original_cwd=original_cwd,
            )
            return result

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
        parser.add_argument("--pair", help="Optional currency pair label used for artefact routing")
        parser.add_argument("--horizon", help="Optional horizon label used for artefact routing")
        args = parser.parse_args()

        run_interpret_command(
            model_module=args.model_module,
            model_factory=args.model_factory,
            events_path=args.events,
            output_dir=args.output_dir,
            seed=args.seed,
            limit=args.limit,
            device=args.device,
            pair=args.pair,
            horizon=args.horizon,
        )
        return

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--multirun", action="store_true", help="launch repeated runs across seeds")
    parser.add_argument(
        "--benchmark-smoke",
        action="store_true",
        help="capture lightweight training/inference benchmarks after training",
    )
    parser.add_argument(
        "--benchmark-full",
        action="store_true",
        help="capture extended training/inference benchmarks after training",
    )
    args, remaining = parser.parse_known_args()

    if args.benchmark_smoke and args.benchmark_full:
        raise SystemExit("--benchmark-smoke and --benchmark-full cannot be used together")
    if args.benchmark_smoke:
        _set_benchmark_mode("smoke")
    elif args.benchmark_full:
        _set_benchmark_mode("full")
    else:
        _set_benchmark_mode(None)

    sys.argv = [sys.argv[0]] + remaining
    if args.multirun:
        _hydra_multirun()
    else:
        _hydra_single()


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    main()
