"""Aggregation helpers for collating multi-run experiment outputs."""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping


LOGGER = logging.getLogger(__name__)

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
    filtered = [float(v) for v in values if not math.isnan(float(v))]
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
        t_critical = _student_t_critical(count - 1)
        ci_radius = t_critical * std / math.sqrt(count)
        effect_size = mean / std if std > 0 else float("nan")
    else:
        std = 0.0
        ci_radius = 0.0
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


def _infer_artifacts_root(path: Path) -> Path | None:
    for parent in [path, *path.parents]:
        if parent.name == "artifacts":
            return parent
    return None


def _resolve_destination(run_root: Path, aggregates_root: Path | None) -> Path:
    artifacts_root = _infer_artifacts_root(run_root)
    if artifacts_root is None:
        destination_root = aggregates_root or (run_root / "aggregates")
        return destination_root

    base = artifacts_root / "runs"
    try:
        relative = run_root.relative_to(base)
    except ValueError:
        relative = run_root.name

    destination_root = aggregates_root or (artifacts_root / "aggregates")
    destination = destination_root / relative
    return destination


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        LOGGER.exception("Failed to parse JSON from %s", path)
        return {}


def _discover_seed_dirs(run_root: Path) -> list[Path]:
    return sorted(dir_path for dir_path in run_root.glob("seed-*") if dir_path.is_dir())


def discover_run_roots(runs_root: Path) -> list[Path]:
    """Locate multirun directories that contain aggregated metadata."""

    if not runs_root.exists():
        return []
    run_roots: set[Path] = set()
    for metadata_path in runs_root.glob("**/metadata.json"):
        parent = metadata_path.parent
        if parent.name.startswith("seed-"):
            continue
        run_roots.add(parent)
    return sorted(run_roots)


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        wrote_row = False
        for row in rows:
            writer.writerow(row)
            wrote_row = True
        if not wrote_row:
            writer.writerow({key: "n/a" for key in fieldnames})
    return path


def _extract_dataset(metadata: Mapping[str, object] | None) -> MutableMapping[str, object]:
    if not metadata:
        return {}
    dataset = metadata.get("dataset")
    if isinstance(dataset, Mapping):
        return dict(dataset)
    return {}


def collate_run_group(run_root: Path, *, aggregates_root: Path | None = None) -> dict[str, Path]:
    """Collate per-seed artifacts under ``run_root`` into aggregate CSV outputs."""

    seed_dirs = _discover_seed_dirs(run_root)
    per_seed_metrics: list[dict[str, float]] = []
    per_seed_metadata: list[dict[str, object]] = []

    for seed_dir in seed_dirs:
        metrics = _load_json(seed_dir / "metrics.json")
        if metrics:
            per_seed_metrics.append({str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        metadata = _load_json(seed_dir / "metadata.json")
        metadata.setdefault("seed", seed_dir.name.split("-", 1)[-1])
        per_seed_metadata.append(metadata)

    metric_names = sorted({name for record in per_seed_metrics for name in record.keys()})
    aggregated: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = [record.get(name, float("nan")) for record in per_seed_metrics]
        aggregated[name] = _compute_stats(values)

    if not aggregated:
        aggregated["placeholder"] = {
            "mean": 0.0,
            "std": 0.0,
            "ci95": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "n": float(len(per_seed_metrics)),
            "cohens_d": float("nan"),
        }

    aggregate_rows = []
    for metric, stats in aggregated.items():
        row = {"metric": metric}
        row.update(stats)
        aggregate_rows.append(row)

    destination = _resolve_destination(run_root, aggregates_root)
    outputs: dict[str, Path] = {}
    aggregate_path = destination / "aggregate.csv"
    outputs["aggregate"] = _write_csv(
        aggregate_path,
        ["metric", "mean", "std", "ci95", "ci95_low", "ci95_high", "n", "cohens_d"],
        aggregate_rows,
    )

    calibration_rows = []
    for record in per_seed_metadata:
        seed_value = record.get("seed")
        metrics = record.get("metrics")
        if isinstance(metrics, Mapping):
            for metric_name, value in metrics.items():
                calibration_rows.append(
                    {
                        "seed": seed_value,
                        "metric": metric_name,
                        "value": value,
                        "aggregate_mean": aggregated.get(metric_name, {}).get("mean"),
                    }
                )
    outputs["calibration"] = _write_csv(
        destination / "calibration.csv",
        ["seed", "metric", "value", "aggregate_mean"],
        calibration_rows,
    )

    aggregate_metadata = _load_json(run_root / "metadata.json")
    dataset_info = _extract_dataset(aggregate_metadata)
    if not dataset_info:
        for record in per_seed_metadata:
            dataset_info = _extract_dataset(record)
            if dataset_info:
                break
    pair_label = str(dataset_info.get("pair", "unknown"))
    horizon_label = str(dataset_info.get("horizon", dataset_info.get("horizon_steps", "unknown")))

    artifacts_root = _infer_artifacts_root(run_root)
    if artifacts_root is not None:
        try:
            relative = run_root.relative_to(artifacts_root / "runs")
            model_label = relative.parts[0] if relative.parts else "model"
        except ValueError:
            model_label = "model"
    else:
        model_label = "model"

    dm_rows = [
        {
            "pair": pair_label,
            "horizon": horizon_label,
            "model_a": model_label,
            "model_b": model_label,
            "statistic": 0.0,
            "p_value": 1.0,
            "notes": "placeholder",
        }
    ]
    outputs["dm_table"] = _write_csv(
        destination / "dm_table.csv",
        ["pair", "horizon", "model_a", "model_b", "statistic", "p_value", "notes"],
        dm_rows,
    )

    spa_rows = [
        {
            "pair": pair_label,
            "horizon": horizon_label,
            "model": model_label,
            "p_value": 1.0,
            "selected": True,
            "notes": "placeholder",
        }
    ]
    outputs["spa_table"] = _write_csv(
        destination / "spa_table.csv",
        ["pair", "horizon", "model", "p_value", "selected", "notes"],
        spa_rows,
    )

    mcs_rows = [
        {
            "pair": pair_label,
            "horizon": horizon_label,
            "model": model_label,
            "included": True,
            "alpha": 0.05,
            "notes": "placeholder",
        }
    ]
    outputs["mcs_table"] = _write_csv(
        destination / "mcs_table.csv",
        ["pair", "horizon", "model", "included", "alpha", "notes"],
        mcs_rows,
    )

    return outputs


__all__ = ["collate_run_group", "discover_run_roots"]
