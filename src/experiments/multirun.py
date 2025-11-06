"""Utilities for launching repeated training runs with different seeds."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, TYPE_CHECKING

try:  # pragma: no cover - optional dependency for precise statistics
    from scipy import stats as _scipy_stats
except ModuleNotFoundError:  # pragma: no cover - runtime fallback when scipy missing
    _scipy_stats = None

if TYPE_CHECKING:  # pragma: no cover - type-checking support only
    from src.training.engine import TrainingSummary

from src.utils.artifacts import build_run_metadata, compute_dataset_checksums


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


_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
    40: 2.021,
    60: 2.000,
    80: 1.990,
    100: 1.984,
    120: 1.980,
}


def _student_t_critical(df: int) -> float:
    if df <= 0:
        return float("nan")
    if _scipy_stats is not None:  # pragma: no cover - exercised when scipy available
        return float(_scipy_stats.t.ppf(0.975, df))
    if df in _T_CRITICAL_95:
        return _T_CRITICAL_95[df]
    keys = sorted(_T_CRITICAL_95)
    lower = max((k for k in keys if k < df), default=keys[0])
    upper = min((k for k in keys if k > df), default=keys[-1])
    if lower == upper:
        return _T_CRITICAL_95[lower]
    lower_val = _T_CRITICAL_95[lower]
    upper_val = _T_CRITICAL_95[upper]
    span = upper - lower
    proportion = (df - lower) / span
    return lower_val + (upper_val - lower_val) * proportion


def _compute_stats(values: Iterable[float]) -> dict[str, float]:
    filtered = [v for v in values if not math.isnan(v)]
    if not filtered:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci95": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "n": 0.0,
            "cohens_d": float("nan"),
        }

    count = len(filtered)
    mean = sum(filtered) / count
    if count > 1:
        variance = sum((v - mean) ** 2 for v in filtered) / (count - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    if count > 1:
        t_critical = _student_t_critical(count - 1)
        ci_radius = t_critical * std / math.sqrt(count)
    else:
        ci_radius = 0.0

    if std > 0:
        effect_size = mean / std
    else:
        effect_size = float("nan")

    return {
        "mean": mean,
        "std": std,
        "ci95": ci_radius,
        "ci95_low": mean - ci_radius,
        "ci95_high": mean + ci_radius,
        "n": float(count),
        "cohens_d": effect_size,
    }


def _write_run_artifacts(output_dir: Path, result: RunResult) -> tuple[dict[str, float], dict[str, object]]:
    run_dir = output_dir / f"seed-{result.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = _extract_summary_metrics(result.summary)
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    artifact_index = {
        "metrics": metrics_path.name,
        "metadata": "metadata.json",
        "manifest": "manifest.json",
    }

    run_metadata = dict(result.metadata)
    dataset_meta = dict(run_metadata.get("dataset") or {})
    dataset_checksums = compute_dataset_checksums(dataset_meta)
    metadata_payload = build_run_metadata(
        run_metadata,
        seed=result.seed,
        device=result.summary.device,
        artifact_index=artifact_index,
        dataset_checksums=dataset_checksums,
    )

    metadata_path = run_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_payload, handle, indent=2, sort_keys=True)

    return metrics, metadata_payload


def _write_summary(output_dir: Path, aggregated: dict[str, dict[str, float]]) -> None:
    fieldnames = ["metric", "mean", "std", "ci95", "ci95_low", "ci95_high", "n", "cohens_d"]
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
    run_records: list[dict[str, object]] = []

    representative_metadata: dict[str, object] | None = None

    for seed in seeds:
        result = runner(seed)
        metrics, run_metadata = _write_run_artifacts(output_dir, result)
        if representative_metadata is None:
            representative_metadata = dict(run_metadata)
        for metric_name, value in metrics.items():
            collected_metrics.setdefault(metric_name, []).append(value)
        run_records.append({"seed": seed, "metrics": metrics, "metadata": run_metadata})

    for metric_name, values in collected_metrics.items():
        aggregated[metric_name] = _compute_stats(values)

    _write_summary(output_dir, aggregated)

    metadata_blob = dict(base_metadata or {})
    if representative_metadata:
        metadata_blob.setdefault("deterministic_flags", representative_metadata.get("deterministic_flags"))
        metadata_blob.setdefault("device", representative_metadata.get("device"))
        metadata_blob.setdefault("hardware", representative_metadata.get("hardware"))
        metadata_blob.setdefault("git_sha", representative_metadata.get("git_sha"))
    metadata_blob["seeds"] = list(seeds)
    metadata_blob["runs"] = len(seeds)
    metadata_blob["metrics"] = aggregated
    metadata_blob["run_records"] = run_records
    metadata_blob["artifacts"] = {
        "metadata": "metadata.json",
        "summary": "summary.csv",
        "runs": [f"seed-{seed}" for seed in seeds],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata_blob, handle, indent=2, sort_keys=True)

    return aggregated
