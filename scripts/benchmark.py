"""Benchmark harness for training/inference throughput and memory usage."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import psutil
import torch
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - environment shim
    sys.path.insert(0, str(REPO_ROOT))

from src.cli import (
    _build_data_config,
    _build_model_config,
    _build_trainer_config,
    _prepare_metadata,
)
from src.data.dataset import create_dataloaders, prepare_datasets
from src.models.forecasting import ModelConfig, TemporalForecastingModel
from src.training.engine import TrainerConfig, train
from src.utils.repro import get_deterministic_flags, hash_config, seed_everything


@dataclass
class TrainingMetrics:
    """Summary of the training benchmark run."""

    elapsed_seconds: float
    epochs: int
    samples_processed: int
    throughput_samples_per_sec: float
    cpu_rss_mb: float
    cpu_maxrss_mb: float
    gpu_peak_mb: float | None


@dataclass
class InferenceMetrics:
    """Summary statistics for the inference benchmark run."""

    batch_size: int
    latency_ms: List[float]
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    cpu_rss_mb: float
    gpu_peak_mb: float | None


@dataclass
class BenchmarkThresholds:
    """Thresholds used to guard benchmarks in CI."""

    min_train_throughput: float | None = None
    max_inference_latency_ms: float | None = None
    max_train_gpu_mb: float | None = None
    max_inference_gpu_mb: float | None = None


@dataclass
class BenchmarkResult:
    """Container combining benchmark metrics and metadata."""

    training: TrainingMetrics
    inference: InferenceMetrics
    metadata: Mapping[str, object]
    environment: Mapping[str, object]
    config_path: str
    config_hash: str
    verification_hash: str


def _quantile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def _capture_command_to_file(path: Path, command: List[str]) -> tuple[str, bool]:
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        payload = completed.stdout
        success = True
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        payload = f"# Failed to execute {' '.join(command)}\n# {exc}\n"
        success = False
    path.write_text(payload)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest, success


def capture_environment_lockfiles(output_dir: Path) -> Mapping[str, object]:
    """Write environment lockfiles and return metadata for verification."""

    env_dir = output_dir / "env"
    env_dir.mkdir(parents=True, exist_ok=True)

    lock_specs = {
        "conda": {
            "command": ["conda", "env", "export", "--no-builds"],
            "filename": "conda-environment.yml",
        },
        "pip": {
            "command": [sys.executable, "-m", "pip", "freeze"],
            "filename": "pip-freeze.txt",
        },
        "cuda": {
            "command": ["nvidia-smi"],
            "filename": "cuda-drivers.txt",
        },
    }

    lockfile_records: list[dict[str, object]] = []
    for name, spec in lock_specs.items():
        path = env_dir / spec["filename"]
        digest, success = _capture_command_to_file(path, spec["command"])
        lockfile_records.append(
            {
                "name": name,
                "path": str(path),
                "sha256": digest,
                "command": spec["command"],
                "success": success,
            }
        )

    combined_hash = hashlib.sha256(
        "".join(sorted(record["sha256"] for record in lockfile_records)).encode("utf-8")
    ).hexdigest()

    return {
        "lockfiles": lockfile_records,
        "combined_hash": combined_hash,
        "python": sys.version,
        "torch_version": torch.__version__,
    }


def _as_mb(value: float) -> float:
    return float(value) / 1024.0 / 1024.0


def _current_rss_mb(process: psutil.Process) -> float:
    return _as_mb(process.memory_info().rss)


def _current_maxrss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _prepare_training_objects(
    cfg: DictConfig, project_root: Path
) -> Tuple[ModelConfig, TrainerConfig, Dict[str, object]]:
    data_cfg = _build_data_config(cfg.data, project_root)
    datasets = prepare_datasets(data_cfg)
    _, window = next(iter(datasets.items()))
    dataloaders = create_dataloaders(window, data_cfg)

    model_cfg = _build_model_config(cfg.model, len(data_cfg.feature_columns), data_cfg.time_steps)
    trainer_cfg = _build_trainer_config(cfg.training)
    return model_cfg, trainer_cfg, dataloaders


def _run_training_epoch(
    model_cfg: ModelConfig,
    trainer_cfg: TrainerConfig,
    dataloaders: Mapping[str, object],
    metadata: Mapping[str, object],
) -> Tuple[TrainingMetrics, TemporalForecastingModel, str | None]:
    model = TemporalForecastingModel(model_cfg)
    trainer_cfg = dataclasses.replace(trainer_cfg)

    process = psutil.Process()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_rss = _current_rss_mb(process)
    start = time.perf_counter()
    summary = train(model, dataloaders, trainer_cfg, metadata=dict(metadata))
    elapsed = time.perf_counter() - start

    train_loader = dataloaders.get("train")
    samples = trainer_cfg.epochs * len(train_loader.dataset) if train_loader else 0
    throughput = samples / elapsed if elapsed else float("inf")

    gpu_peak = _as_mb(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None

    metrics = TrainingMetrics(
        elapsed_seconds=elapsed,
        epochs=trainer_cfg.epochs,
        samples_processed=samples,
        throughput_samples_per_sec=throughput,
        cpu_rss_mb=max(start_rss, _current_rss_mb(process)),
        cpu_maxrss_mb=_current_maxrss_mb(),
        gpu_peak_mb=gpu_peak,
    )
    return metrics, model, getattr(summary, "device", None)


def _run_inference_latency(
    model: TemporalForecastingModel,
    dataloaders,
    warmup_iters: int,
    measure_iters: int,
    device: torch.device,
) -> InferenceMetrics:
    model.eval()
    model.to(device)

    iterator = iter(dataloaders["test"])
    try:
        batch_inputs, _ = next(iterator)
    except StopIteration:
        iterator = iter(dataloaders["val"])
        batch_inputs, _ = next(iterator)

    batch_inputs = batch_inputs.to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(max(0, warmup_iters)):
            model(batch_inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(measure_iters):
            start = time.perf_counter()
            model(batch_inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)

    process = psutil.Process()
    gpu_peak = _as_mb(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None

    latency_ms = [lat * 1000.0 for lat in latencies]
    mean_latency = statistics.mean(latency_ms) if latency_ms else float("nan")
    median_latency = _quantile(latency_ms, 0.5)
    p95_latency = _quantile(latency_ms, 0.95)

    return InferenceMetrics(
        batch_size=batch_inputs.shape[0],
        latency_ms=latency_ms,
        mean_latency_ms=mean_latency,
        median_latency_ms=median_latency,
        p95_latency_ms=p95_latency,
        cpu_rss_mb=_current_rss_mb(process),
        gpu_peak_mb=gpu_peak,
    )


def run_benchmark(
    cfg_path: Path,
    output_dir: Path,
    train_warmup_epochs: int,
    inference_warmup_iters: int,
    inference_measure_iters: int,
) -> BenchmarkResult:
    cfg = OmegaConf.load(cfg_path)
    project_root = Path.cwd()

    metadata = _prepare_metadata(cfg)
    metadata = dict(metadata)
    metadata["deterministic_flags"] = get_deterministic_flags()

    seed = int(cfg.seed)
    seed_everything(seed)
    model_cfg, trainer_cfg, dataloaders = _prepare_training_objects(cfg, project_root)

    if train_warmup_epochs > 0:
        warm_cfg = dataclasses.replace(trainer_cfg, epochs=train_warmup_epochs)
        seed_everything(seed)
        train(TemporalForecastingModel(model_cfg), dataloaders, warm_cfg, metadata=dict(metadata))

    seed_everything(seed)
    training_metrics, model, device_str = _run_training_epoch(
        model_cfg, trainer_cfg, dataloaders, metadata
    )

    if device_str and device_str != "auto":
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference_metrics = _run_inference_latency(
        model,
        dataloaders,
        warmup_iters=inference_warmup_iters,
        measure_iters=inference_measure_iters,
        device=device,
    )

    environment = capture_environment_lockfiles(output_dir)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_path": str(cfg_path),
        "config_hash": hash_config(cfg),
        "metadata": metadata,
        "training": dataclasses.asdict(training_metrics),
        "inference": dataclasses.asdict(inference_metrics),
        "environment": environment,
    }
    verification_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return BenchmarkResult(
        training=training_metrics,
        inference=inference_metrics,
        metadata=metadata,
        environment=environment,
        config_path=str(cfg_path),
        config_hash=payload["config_hash"],
        verification_hash=verification_hash,
    )


def enforce_thresholds(result: BenchmarkResult, thresholds: BenchmarkThresholds) -> None:
    """Raise if the benchmark metrics fall outside allowed ranges."""

    failures: list[str] = []
    if (
        thresholds.min_train_throughput is not None
        and result.training.throughput_samples_per_sec < thresholds.min_train_throughput
    ):
        failures.append(
            f"Training throughput {result.training.throughput_samples_per_sec:.2f} < minimum {thresholds.min_train_throughput:.2f}"
        )
    if (
        thresholds.max_inference_latency_ms is not None
        and result.inference.mean_latency_ms > thresholds.max_inference_latency_ms
    ):
        failures.append(
            f"Inference latency {result.inference.mean_latency_ms:.2f} ms > maximum {thresholds.max_inference_latency_ms:.2f} ms"
        )
    if (
        thresholds.max_train_gpu_mb is not None
        and result.training.gpu_peak_mb is not None
        and result.training.gpu_peak_mb > thresholds.max_train_gpu_mb
    ):
        failures.append(
            f"Training GPU memory {result.training.gpu_peak_mb:.2f} MB > maximum {thresholds.max_train_gpu_mb:.2f} MB"
        )
    if (
        thresholds.max_inference_gpu_mb is not None
        and result.inference.gpu_peak_mb is not None
        and result.inference.gpu_peak_mb > thresholds.max_inference_gpu_mb
    ):
        failures.append(
            f"Inference GPU memory {result.inference.gpu_peak_mb:.2f} MB > maximum {thresholds.max_inference_gpu_mb:.2f} MB"
        )

    if failures:
        raise RuntimeError("Benchmark thresholds violated:\n" + "\n".join(failures))


def _write_summary(result: BenchmarkResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": timestamp,
        "config_path": result.config_path,
        "config_hash": result.config_hash,
        "training": dataclasses.asdict(result.training),
        "inference": dataclasses.asdict(result.inference),
        "metadata": result.metadata,
        "environment": result.environment,
        "verification_hash": result.verification_hash,
    }

    summary_path = output_dir / f"summary_{timestamp}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    latest_path = output_dir / "latest.json"
    latest_path.write_text(summary_path.read_text())
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark training and inference performance")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmark"))
    parser.add_argument("--train-warmup", type=int, default=1)
    parser.add_argument("--inference-warmup", type=int, default=5)
    parser.add_argument("--inference-runs", type=int, default=20)
    parser.add_argument("--min-train-throughput", type=float)
    parser.add_argument("--max-inference-latency-ms", type=float)
    parser.add_argument("--max-train-gpu-mb", type=float)
    parser.add_argument("--max-inference-gpu-mb", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = run_benchmark(
        args.config,
        args.output_dir,
        train_warmup_epochs=max(0, args.train_warmup),
        inference_warmup_iters=max(0, args.inference_warmup),
        inference_measure_iters=max(1, args.inference_runs),
    )

    summary_path = _write_summary(result, args.output_dir)
    print(f"Benchmark summary written to {summary_path}")

    thresholds = BenchmarkThresholds(
        min_train_throughput=args.min_train_throughput,
        max_inference_latency_ms=args.max_inference_latency_ms,
        max_train_gpu_mb=args.max_train_gpu_mb,
        max_inference_gpu_mb=args.max_inference_gpu_mb,
    )
    try:
        enforce_thresholds(result, thresholds)
    except RuntimeError as exc:
        print(str(exc))
        raise


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
