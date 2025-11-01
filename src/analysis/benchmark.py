"""Compute benchmarking utilities for training and inference workloads."""

from __future__ import annotations

import argparse
import logging
import os
import platform
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

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
    latency_ms: float
    max_memory_mb: Optional[float]


@dataclass
class BenchmarkReport:
    """Full benchmark output containing metadata and metrics."""

    hardware: HardwareSpec
    metrics: BenchmarkMetrics


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
) -> BenchmarkReport:
    """Benchmark a model over an iterable of batches."""

    if mode not in {"inference", "training"}:
        raise ValueError("mode must be 'inference' or 'training'")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training = mode == "training"
    model = model.to(device)
    model.train(training)
    if training and (loss_fn is None or optimizer is None):
        raise ValueError("Training benchmarks require both a loss function and an optimizer")

    iterator = _cycle(dataloader)

    # Warm-up
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

    # Measurement
    torch.cuda.empty_cache() if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:  # pragma: no cover - optional dependency
        import psutil

        process = psutil.Process()
        start_rss = process.memory_info().rss
    except Exception:  # pragma: no cover - best effort
        process = None  # type: ignore[assignment]
        start_rss = None

    total_samples = 0
    start_time = time.perf_counter()

    for batch in islice(iterator, measure_steps):
        inputs, targets = _prepare_batch(batch, device)
        batch_size = _infer_batch_size(inputs)
        total_samples += batch_size
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

    elapsed = time.perf_counter() - start_time

    max_memory_mb: Optional[float]
    if device.type == "cuda":
        max_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        if process is not None and start_rss is not None:
            max_memory_mb = max(process.memory_info().rss - start_rss, 0) / (1024**2)
        else:
            max_memory_mb = None

    throughput = total_samples / elapsed if elapsed > 0 else 0.0
    latency_ms = (elapsed / measure_steps) * 1000 if measure_steps else 0.0

    metrics = BenchmarkMetrics(
        mode=mode,
        warmup_steps=warmup_steps,
        measured_steps=measure_steps,
        batch_size=int(total_samples / measure_steps) if measure_steps else 0,
        throughput_samples_per_sec=throughput,
        latency_ms=latency_ms,
        max_memory_mb=max_memory_mb,
    )
    hardware = _gather_hardware_spec(device)
    LOGGER.info(
        "Benchmark complete - mode: %s | throughput: %.2f samples/s | latency: %.2f ms",
        mode,
        throughput,
        latency_ms,
    )
    return BenchmarkReport(hardware=hardware, metrics=metrics)


def save_report(report: BenchmarkReport, output_dir: Path) -> Path:
    """Persist a benchmark report to disk as JSON."""

    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "hardware": report.hardware.__dict__,
        "metrics": report.metrics.__dict__,
    }
    path = output_dir / "benchmark.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


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
    save_path = save_report(report, args.output)
    LOGGER.info("Benchmark report saved to %s", save_path)
