"""Utilities for launching repeated training runs with different seeds."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type-checking support only
    from src.training.engine import TrainingSummary


@dataclass
class RunResult:
    seed: int
    summary: "TrainingSummary"
    metadata: dict[str, object]


def _extract_summary_metrics(summary: "TrainingSummary") -> dict[str, float]:
    epochs = list(summary.epochs)
    final_val_loss = epochs[-1].val_loss if epochs else float("nan")
    final_val_mae = epochs[-1].val_mae if epochs else float("nan")
    return {
        "best_val_loss": summary.best_val_loss,
        "final_val_loss": final_val_loss,
        "final_val_mae": final_val_mae,
    }


def _compute_stats(values: Iterable[float]) -> tuple[float, float, float]:
    filtered = [v for v in values if not math.isnan(v)]
    if not filtered:
        return float("nan"), float("nan"), float("nan")

    mean = sum(filtered) / len(filtered)
    if len(filtered) > 1:
        variance = sum((v - mean) ** 2 for v in filtered) / (len(filtered) - 1)
        std = math.sqrt(variance)
        ci95 = 1.96 * std / math.sqrt(len(filtered))
    else:
        std = 0.0
        ci95 = 0.0

    return mean, std, ci95


def _write_run_artifacts(output_dir: Path, result: RunResult) -> dict[str, float]:
    run_dir = output_dir / f"seed-{result.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = _extract_summary_metrics(result.summary)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    run_metadata = dict(result.metadata)
    run_metadata["seed"] = result.seed
    run_metadata.setdefault("device", result.summary.device)
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2, sort_keys=True)

    return metrics


def _write_summary(output_dir: Path, aggregated: dict[str, dict[str, float]]) -> None:
    fieldnames = ["metric", "mean", "std", "ci95"]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric, stats in aggregated.items():
            row = {"metric": metric}
            row.update(stats)
            writer.writerow(row)


def run_multirun(
    seeds: Sequence[int],
    output_dir: Path,
    runner: Callable[[int], RunResult],
    base_metadata: dict[str, object] | None = None,
) -> dict[str, dict[str, float]]:
    """Execute `runner` for each seed and persist per-run artifacts and aggregates."""

    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated: dict[str, dict[str, float]] = {}
    collected_metrics: dict[str, list[float]] = {}

    representative_metadata: dict[str, object] | None = None

    for seed in seeds:
        result = runner(seed)
        metrics = _write_run_artifacts(output_dir, result)
        if representative_metadata is None:
            representative_metadata = dict(result.metadata)
        for metric_name, value in metrics.items():
            collected_metrics.setdefault(metric_name, []).append(value)

    for metric_name, values in collected_metrics.items():
        mean, std, ci95 = _compute_stats(values)
        aggregated[metric_name] = {"mean": mean, "std": std, "ci95": ci95}

    _write_summary(output_dir, aggregated)

    metadata_blob = dict(base_metadata or {})
    if representative_metadata:
        metadata_blob.setdefault("deterministic_flags", representative_metadata.get("deterministic_flags"))
        metadata_blob.setdefault("device", representative_metadata.get("device"))
    metadata_blob["seeds"] = list(seeds)
    metadata_blob["runs"] = len(seeds)
    metadata_blob["metrics"] = aggregated
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata_blob, handle, indent=2, sort_keys=True)

    return aggregated
