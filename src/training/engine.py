"""Training utilities consolidated from the iterative prototypes in `v_08`."""
from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, TYPE_CHECKING

from src.utils.manifest import write_manifest

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

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        processed = 0

        for step, (inputs, targets) in enumerate(train_loader, start=1):
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

    return TrainingSummary(
        epochs=history,
        best_val_loss=best_val_loss,
        device=str(device),
        benchmarks={},
    )
