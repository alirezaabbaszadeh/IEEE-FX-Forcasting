"""Compute benchmarking utilities for training and inference workloads."""

from __future__ import annotations

import argparse
import logging
import os
import platform
import time
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from statistics import mean, stdev
from typing import Callable, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency for statistics when available
    from statistics import fmean as _fmean
except ImportError:  # pragma: no cover - Python < 3.8 fallback
    _fmean = mean

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

LOGGER = logging.getLogger(__name__)


@dataclass
class HardwareSpec:
    """Snapshot of the machine that produced a benchmark measurement."""

    device: str
    cpu: str
    cpu_count: int
    memory_gb: Optional[float]
    gpu: Optional[str]
    cuda_capability: Optional[str]


@dataclass
class BenchmarkMetrics:
    """Summary statistics for a benchmark run."""

    mode: str
    warmup_steps: int
    measured_steps: int
    batch_size: int
    throughput_samples_per_sec: float
    warmup_time_s: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_std_ms: float
    latency_ms_samples: Tuple[float, ...]
    max_memory_mb: Optional[float]
    cpu_rss_delta_mb: Optional[float]
    cpu_rss_end_mb: Optional[float]


@dataclass
class BenchmarkSettings:
    """Configuration used when capturing a benchmark."""

    device: str
    dataloader: Optional[str]
    mode: str
    warmup_steps: int
    measured_steps: int
    batch_size: int


@dataclass
class BenchmarkReport:
    """Full benchmark output containing metadata and metrics."""

    hardware: HardwareSpec
    metrics: BenchmarkMetrics
    settings: BenchmarkSettings


@dataclass
class SavedReportPaths:
    """Locations where benchmark artefacts were persisted."""

    json_path: Path
    csv_path: Path


def _gather_hardware_spec(device: torch.device) -> HardwareSpec:
    cpu = platform.processor() or platform.machine()
    cpu_count = os.cpu_count() or 1
    memory_gb: Optional[float] = None
    try:  # pragma: no cover - optional dependency
        import psutil

        memory_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:  # pragma: no cover - best effort
        memory_gb = None

    gpu_name: Optional[str] = None
    capability: Optional[str] = None
    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        gpu_name = props.name
        capability = f"{props.major}.{props.minor}"
    return HardwareSpec(
        device=str(device),
        cpu=cpu,
        cpu_count=cpu_count,
        memory_gb=memory_gb,
        gpu=gpu_name,
        cuda_capability=capability,
    )


def _cycle(iterator: Iterable) -> Iterator:
    while True:
        for item in iterator:
            yield item


def _prepare_batch(batch: object, device: torch.device) -> Tuple[Tensor, Optional[Tensor]]:
    if isinstance(batch, Mapping):
        inputs = batch.get("inputs")
        targets = batch.get("targets")
        if inputs is None:
            raise ValueError("Mapping batches must contain an 'inputs' key")
        inputs = inputs.to(device)
        targets = targets.to(device) if isinstance(targets, Tensor) else targets
        return inputs, targets
    if isinstance(batch, Sequence) and batch and not isinstance(batch, (str, bytes)):
        inputs = batch[0].to(device)
        target = batch[1].to(device) if len(batch) > 1 and isinstance(batch[1], Tensor) else None
        return inputs, target
    if isinstance(batch, Tensor):
        return batch.to(device), None
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def _infer_batch_size(batch: Tensor) -> int:
    if batch.ndim == 0:
        return 1
    return int(batch.shape[0])


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def benchmark_model(
    model: nn.Module,
    dataloader: Iterable,
    *,
    mode: str = "inference",
    warmup_steps: int = 2,
    measure_steps: int = 10,
    device: Optional[torch.device] = None,
    loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    optimizer: Optional[Optimizer] = None,
    dataloader_label: Optional[str] = None,
) -> BenchmarkReport:
    """Benchmark a model over an iterable of batches."""

    if mode not in {"inference", "training"}:
        raise ValueError("mode must be 'inference' or 'training'")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training = mode == "training"
    model = model.to(device)
    previous_mode = model.training
    model.train(training)
    if training and (loss_fn is None or optimizer is None):
        raise ValueError("Training benchmarks require both a loss function and an optimizer")

    iterator = _cycle(dataloader)

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)

    warmup_start = time.perf_counter()
    for batch in islice(iterator, warmup_steps):
        inputs, targets = _prepare_batch(batch, device)
        if training:
            assert loss_fn is not None and optimizer is not None  # for type checkers
            if targets is None:
                raise ValueError("Training benchmarks require batches with targets")
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model(inputs)

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    warmup_time = time.perf_counter() - warmup_start

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    try:  # pragma: no cover - optional dependency
        import psutil

        process = psutil.Process()
        start_rss = process.memory_info().rss
    except Exception:  # pragma: no cover - best effort
        process = None  # type: ignore[assignment]
        start_rss = None

    total_samples = 0
    latencies_ms: list[float] = []

    for batch in islice(iterator, measure_steps):
        inputs, targets = _prepare_batch(batch, device)
        batch_size = _infer_batch_size(inputs)
        total_samples += batch_size
        step_start = time.perf_counter()
        if training:
            assert loss_fn is not None and optimizer is not None
            if targets is None:
                raise ValueError("Training benchmarks require batches with targets")
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model(inputs)

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

        latencies_ms.append((time.perf_counter() - step_start) * 1000)

    if device.type == "cuda" and torch.cuda.is_available():
        max_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        if process is not None and start_rss is not None:
            max_memory_mb = max(process.memory_info().rss - start_rss, 0) / (1024**2)
        else:
            max_memory_mb = None

    if process is not None and start_rss is not None:
        end_rss = process.memory_info().rss
        cpu_rss_delta_mb = max(end_rss - start_rss, 0) / (1024**2)
        cpu_rss_end_mb = end_rss / (1024**2)
    else:
        cpu_rss_delta_mb = None
        cpu_rss_end_mb = None

    measured_batch_size = int(total_samples / measure_steps) if measure_steps else 0
    total_elapsed_s = sum(latencies_ms) / 1000.0 if latencies_ms else 0.0
    throughput = total_samples / total_elapsed_s if total_elapsed_s > 0 else 0.0

    latency_mean = _fmean(latencies_ms) if latencies_ms else 0.0
    latency_std = stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    latency_p50 = _percentile(latencies_ms, 0.5)
    latency_p90 = _percentile(latencies_ms, 0.9)
    latency_p95 = _percentile(latencies_ms, 0.95)
    latency_p99 = _percentile(latencies_ms, 0.99)
    latency_min = min(latencies_ms) if latencies_ms else 0.0
    latency_max = max(latencies_ms) if latencies_ms else 0.0

    metrics = BenchmarkMetrics(
        mode=mode,
        warmup_steps=warmup_steps,
        measured_steps=measure_steps,
        batch_size=measured_batch_size,
        throughput_samples_per_sec=throughput,
        warmup_time_s=warmup_time,
        latency_mean_ms=latency_mean,
        latency_p50_ms=latency_p50,
        latency_p90_ms=latency_p90,
        latency_p95_ms=latency_p95,
        latency_p99_ms=latency_p99,
        latency_min_ms=latency_min,
        latency_max_ms=latency_max,
        latency_std_ms=latency_std,
        latency_ms_samples=tuple(latencies_ms),
        max_memory_mb=max_memory_mb,
        cpu_rss_delta_mb=cpu_rss_delta_mb,
        cpu_rss_end_mb=cpu_rss_end_mb,
    )
    hardware = _gather_hardware_spec(device)
    settings = BenchmarkSettings(
        device=str(device),
        dataloader=dataloader_label,
        mode=mode,
        warmup_steps=warmup_steps,
        measured_steps=measure_steps,
        batch_size=measured_batch_size,
    )
    LOGGER.info(
        "Benchmark complete - mode: %s | throughput: %.2f samples/s | latency: %.2f ms",
        mode,
        throughput,
        latency_mean,
    )
    model.train(previous_mode)
    return BenchmarkReport(hardware=hardware, metrics=metrics, settings=settings)


def save_report(report: BenchmarkReport, output_dir: Path, *, stem: str = "benchmark") -> SavedReportPaths:
    """Persist a benchmark report to disk as JSON and CSV."""

    import csv
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "hardware": asdict(report.hardware),
        "metrics": asdict(report.metrics),
        "settings": asdict(report.settings),
    }

    json_path = output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    csv_path = output_dir / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "mode",
            "dataloader",
            "warmup_steps",
            "measured_steps",
            "batch_size",
            "warmup_time_s",
            "throughput_samples_per_sec",
            "latency_mean_ms",
            "latency_p50_ms",
            "latency_p90_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "latency_min_ms",
            "latency_max_ms",
            "latency_std_ms",
            "max_memory_mb",
            "cpu_rss_delta_mb",
            "cpu_rss_end_mb",
            "hardware_device",
            "hardware_cpu",
            "hardware_cpu_count",
            "hardware_memory_gb",
            "hardware_gpu",
            "hardware_cuda_capability",
            "latency_ms_samples",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        row = {
            "mode": report.metrics.mode,
            "dataloader": report.settings.dataloader,
            "warmup_steps": report.metrics.warmup_steps,
            "measured_steps": report.metrics.measured_steps,
            "batch_size": report.metrics.batch_size,
            "warmup_time_s": report.metrics.warmup_time_s,
            "throughput_samples_per_sec": report.metrics.throughput_samples_per_sec,
            "latency_mean_ms": report.metrics.latency_mean_ms,
            "latency_p50_ms": report.metrics.latency_p50_ms,
            "latency_p90_ms": report.metrics.latency_p90_ms,
            "latency_p95_ms": report.metrics.latency_p95_ms,
            "latency_p99_ms": report.metrics.latency_p99_ms,
            "latency_min_ms": report.metrics.latency_min_ms,
            "latency_max_ms": report.metrics.latency_max_ms,
            "latency_std_ms": report.metrics.latency_std_ms,
            "max_memory_mb": report.metrics.max_memory_mb,
            "cpu_rss_delta_mb": report.metrics.cpu_rss_delta_mb,
            "cpu_rss_end_mb": report.metrics.cpu_rss_end_mb,
            "hardware_device": report.hardware.device,
            "hardware_cpu": report.hardware.cpu,
            "hardware_cpu_count": report.hardware.cpu_count,
            "hardware_memory_gb": report.hardware.memory_gb,
            "hardware_gpu": report.hardware.gpu,
            "hardware_cuda_capability": report.hardware.cuda_capability,
            "latency_ms_samples": " ".join(f"{value:.6f}" for value in report.metrics.latency_ms_samples),
        }
        writer.writerow(row)

    return SavedReportPaths(json_path=json_path, csv_path=csv_path)


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run a simple benchmark on random data.")
    parser.add_argument("--mode", choices=["inference", "training"], default="inference")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmarks"))
    args = parser.parse_args()

    from torch.utils.data import DataLoader, TensorDataset

    total_batches = args.steps + 2
    dataset = TensorDataset(
        torch.randn(args.batch_size * total_batches, args.seq_len, args.features),
        torch.randn(args.batch_size * total_batches, 1),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    model = nn.Sequential(nn.Flatten(), nn.Linear(args.seq_len * args.features, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    report = benchmark_model(
        model,
        dataloader,
        mode=args.mode,
        warmup_steps=2,
        measure_steps=args.steps,
        loss_fn=loss_fn,
        optimizer=optimizer if args.mode == "training" else None,
    )
    paths = save_report(report, args.output)
    LOGGER.info("Benchmark report saved to %s", paths.json_path)
