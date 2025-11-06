"""Training utilities consolidated from the iterative prototypes in `v_08`."""
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, TYPE_CHECKING

from src.utils.manifest import write_manifest

try:  # pragma: no cover - optional dependency for system telemetry
    import psutil
except ModuleNotFoundError:  # pragma: no cover - graceful degradation when psutil absent
    psutil = None  # type: ignore[assignment]

try:  # pragma: no cover - GPU telemetry is optional
    import pynvml  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - environments without NVML support
    pynvml = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


def _log_training_metadata(metadata: dict[str, object] | None) -> None:
    if not metadata:
        return

    keys_to_highlight = ("config_hash", "git_sha", "hardware")
    for key in keys_to_highlight:
        if key in metadata:
            LOGGER.info("Training metadata - %s: %s", key, metadata[key])

    for key, value in metadata.items():
        if key in keys_to_highlight:
            continue
        LOGGER.debug("Training metadata detail - %s: %s", key, value)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from src.analysis.benchmark import BenchmarkReport
else:  # pragma: no cover - optional dependency handling
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ModuleNotFoundError:  # pragma: no cover - fallback for environments without torch
        torch = None  # type: ignore[assignment]
        nn = None  # type: ignore[assignment]

        class DataLoader:  # type: ignore[override]
            pass

@dataclass
class TrainerConfig:
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: Optional[float] = None
    log_interval: int = 20
    device: str = "auto"


@dataclass
class EpochMetrics:
    train_loss: float
    val_loss: float
    val_mae: float


@dataclass
class TrainingSummary:
    epochs: Iterable[EpochMetrics] = field(default_factory=list)
    best_val_loss: float = float("inf")
    device: str | None = None
    benchmarks: dict[str, "BenchmarkReport"] = field(default_factory=dict)
    compute: "ComputeStats" | None = None


@dataclass
class ComputeStats:
    """Container capturing coarse compute utilisation statistics for a run."""

    wall_time_s: float
    cpu_rss_mb_mean: float | None = None
    cpu_rss_mb_peak: float | None = None
    gpu_utilization_mean: float | None = None
    gpu_utilization_max: float | None = None
    gpu_memory_mb_peak: float | None = None
    samples: int = 0

    def as_dict(self) -> dict[str, float]:
        """Serialise statistics into a CSV/JSON friendly mapping."""

        def _to_float(value: float | None) -> float:
            if value is None:
                return float("nan")
            if isinstance(value, float) and math.isnan(value):
                return float("nan")
            return float(value)

        return {
            "wall_time_s": _to_float(self.wall_time_s),
            "cpu_rss_mb_mean": _to_float(self.cpu_rss_mb_mean),
            "cpu_rss_mb_peak": _to_float(self.cpu_rss_mb_peak),
            "gpu_utilization_mean": _to_float(self.gpu_utilization_mean),
            "gpu_utilization_max": _to_float(self.gpu_utilization_max),
            "gpu_memory_mb_peak": _to_float(self.gpu_memory_mb_peak),
            "samples": float(self.samples),
        }


def _resolve_device(requested: str) -> torch.device:
    if torch is None:  # pragma: no cover - safeguard for optional dependency
        raise ImportError("PyTorch is required to resolve training devices")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _calculate_loss_sum(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    if torch is None:  # pragma: no cover - safeguard for optional dependency
        raise ImportError("PyTorch is required to calculate losses")
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    if len(dataloader.dataset) == 0:
        return float("nan"), float("nan")

    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            mae = torch.mean(torch.abs(outputs - targets))

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_mae += mae.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    return avg_loss, avg_mae


class _ComputeMonitor:
    """Lightweight sampler capturing runtime, memory, and utilisation statistics."""

    _BYTES_TO_MB = 1024 ** 2

    def __init__(self, device: torch.device | None) -> None:
        self._device = device
        self._process = psutil.Process(os.getpid()) if psutil is not None else None
        self._start = time.perf_counter()
        self._cpu_samples: list[float] = []
        self._cpu_peak = 0.0
        self._gpu_util_samples: list[float] = []
        self._gpu_util_peak = 0.0
        self._gpu_mem_peak = 0.0
        self._samples = 0
        self._nvml_handle = None
        self._nvml_initialised = False

        if self._process is not None:
            try:
                self._process.cpu_percent(None)
            except Exception:  # pragma: no cover - defensive against psutil quirks
                LOGGER.debug("psutil.cpu_percent warm-up failed", exc_info=True)

        if (
            pynvml is not None
            and device is not None
            and getattr(device, "type", "cpu") == "cuda"
        ):
            try:
                pynvml.nvmlInit()
                index = device.index if device.index is not None else torch.cuda.current_device()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(int(index))
                self._nvml_initialised = True
            except Exception:  # pragma: no cover - NVML initialisation failures are non-fatal
                LOGGER.debug("NVML initialisation failed", exc_info=True)
                self._nvml_handle = None
                self._nvml_initialised = False

        if torch is not None and device is not None and getattr(device, "type", "cpu") == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:  # pragma: no cover - optional depending on CUDA availability
                LOGGER.debug("Failed to reset CUDA peak memory stats", exc_info=True)

    def record(self) -> None:
        """Capture a snapshot of runtime resource utilisation."""

        if self._process is not None:
            try:
                rss = self._process.memory_info().rss / self._BYTES_TO_MB
            except Exception:  # pragma: no cover - psutil edge cases
                LOGGER.debug("Failed to sample RSS", exc_info=True)
            else:
                self._cpu_samples.append(float(rss))
                self._cpu_peak = max(self._cpu_peak, float(rss))

        if self._nvml_handle is not None and pynvml is not None:
            try:
                utilisation = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            except Exception:  # pragma: no cover - NVML may intermittently fail
                LOGGER.debug("Failed to sample NVML utilisation", exc_info=True)
            else:
                util_gpu = float(utilisation.gpu)
                mem_used = float(memory.used) / self._BYTES_TO_MB
                self._gpu_util_samples.append(util_gpu)
                self._gpu_util_peak = max(self._gpu_util_peak, util_gpu)
                self._gpu_mem_peak = max(self._gpu_mem_peak, mem_used)
        elif (
            torch is not None
            and self._device is not None
            and getattr(self._device, "type", "cpu") == "cuda"
        ):
            try:
                mem_bytes = torch.cuda.max_memory_allocated(self._device)
            except Exception:  # pragma: no cover - optional depending on CUDA runtime
                LOGGER.debug("Failed to sample CUDA memory", exc_info=True)
            else:
                mem_used = float(mem_bytes) / self._BYTES_TO_MB
                self._gpu_mem_peak = max(self._gpu_mem_peak, mem_used)

        self._samples += 1

    def finish(self) -> ComputeStats:
        """Finalize monitoring and return aggregate statistics."""

        wall_time = time.perf_counter() - self._start
        cpu_mean = (
            sum(self._cpu_samples) / len(self._cpu_samples)
            if self._cpu_samples
            else None
        )
        gpu_mean = (
            sum(self._gpu_util_samples) / len(self._gpu_util_samples)
            if self._gpu_util_samples
            else None
        )

        if (
            torch is not None
            and self._device is not None
            and getattr(self._device, "type", "cpu") == "cuda"
        ):
            try:
                mem_bytes = torch.cuda.max_memory_allocated(self._device)
            except Exception:  # pragma: no cover - optional depending on CUDA runtime
                LOGGER.debug("Failed to finalise CUDA peak memory stats", exc_info=True)
            else:
                mem_used = float(mem_bytes) / self._BYTES_TO_MB
                self._gpu_mem_peak = max(self._gpu_mem_peak, mem_used)

        if self._nvml_initialised and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # pragma: no cover - NVML shutdown failures are benign
                LOGGER.debug("NVML shutdown failed", exc_info=True)

        return ComputeStats(
            wall_time_s=float(wall_time),
            cpu_rss_mb_mean=cpu_mean,
            cpu_rss_mb_peak=self._cpu_peak if self._cpu_samples else None,
            gpu_utilization_mean=gpu_mean,
            gpu_utilization_max=self._gpu_util_peak if self._gpu_util_samples else None,
            gpu_memory_mb_peak=self._gpu_mem_peak or None,
            samples=self._samples,
        )


def train(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    cfg: TrainerConfig,
    metadata: Optional[dict[str, object]] = None,
    *,
    manifest_path: Path | None = None,
) -> TrainingSummary:
    """Train a model using mean-squared-error loss and report validation metrics."""

    if torch is None or nn is None:  # pragma: no cover - safeguard for optional dependency
        raise ImportError("PyTorch is required to run training")

    device = _resolve_device(cfg.device)
    LOGGER.info("Using device: %s", device)
    if metadata is None:
        metadata = {}
    metadata.setdefault("device", str(device))
    _log_training_metadata(metadata)
    if manifest_path is not None:
        try:
            write_manifest(manifest_path, metadata)
        except Exception:  # pragma: no cover - defensive: manifest must not break training
            LOGGER.exception("Failed to write training manifest to %s", manifest_path)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    history: list[EpochMetrics] = []
    best_val_loss = float("inf")

    train_loader = dataloaders.get("train")
    val_loader = dataloaders.get("val")

    if train_loader is None:
        raise ValueError("`train` dataloader is required")

    monitor = _ComputeMonitor(device)
    monitor.record()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        processed = 0

        for step, (inputs, targets) in enumerate(train_loader, start=1):
            monitor.record()
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            processed += batch_size

            if cfg.log_interval and step % cfg.log_interval == 0:
                LOGGER.info(
                    "Epoch %d Step %d/%d - loss: %.4f",
                    epoch,
                    step,
                    len(train_loader),
                    loss.item(),
                )

        train_loss = running_loss / processed if processed else float("nan")
        val_loss = float("nan")
        val_mae = float("nan")

        if val_loader is not None and len(val_loader.dataset) > 0:
            monitor.record()
            val_loss, val_mae = _calculate_loss_sum(model, val_loader, criterion, device)
            best_val_loss = min(best_val_loss, val_loss)

        metrics = EpochMetrics(train_loss=train_loss, val_loss=val_loss, val_mae=val_mae)
        history.append(metrics)
        LOGGER.info(
            "Epoch %d completed - train_loss: %.4f, val_loss: %.4f, val_mae: %.4f",
            epoch,
            train_loss,
            val_loss,
            val_mae,
        )

    compute_stats = monitor.finish()

    return TrainingSummary(
        epochs=history,
        best_val_loss=best_val_loss,
        device=str(device),
        benchmarks={},
        compute=compute_stats,
    )
