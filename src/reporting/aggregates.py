"""Aggregation helpers for collating multi-run experiment outputs."""

from __future__ import annotations

import csv
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence


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


def _load_compute(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            row = next(reader, None)
    except (OSError, StopIteration):  # pragma: no cover - defensive file handling
        return {}
    if not row:
        return {}
    record: dict[str, float] = {}
    for key, value in row.items():
        try:
            record[key] = float(value)
        except (TypeError, ValueError):
            continue
    return record


def _extract_baseline_mse(metadata: Mapping[str, object] | None) -> float:
    if not metadata:
        return float("nan")
    baseline = metadata.get("baseline_metrics")
    if not isinstance(baseline, Mapping):
        return float("nan")
    val_metrics = baseline.get("val")
    if isinstance(val_metrics, Mapping):
        value = val_metrics.get("mse") or val_metrics.get("loss")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")
    return float("nan")


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


def _maybe_emit_variant_summary(
    run_root: Path, destination: Path, outputs: dict[str, Path]
) -> None:
    candidates = [run_root / "variants.json", run_root.parent / "variants.json"]
    summary_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if summary_path is None:
        return

    summary = _load_json(summary_path)
    baseline = summary.get("baseline")
    baseline_label = str(baseline) if baseline is not None else None

    # Avoid duplicating outputs when the summary lives alongside sibling variants.
    if summary_path.parent == run_root.parent and baseline_label and run_root.name != baseline_label:
        return

    variants = summary.get("variants")
    if not isinstance(variants, Sequence):
        return

    thresholds = summary.get("thresholds") if isinstance(summary, Mapping) else {}

    rows: list[dict[str, object]] = []
    for entry in variants:
        if not isinstance(entry, Mapping):
            continue
        deltas = entry.get("deltas")
        if not isinstance(deltas, Mapping) or not deltas:
            continue
        variant_name = str(entry.get("name"))
        for metric_name, payload in deltas.items():
            if not isinstance(payload, Mapping):
                continue
            absolute = payload.get("absolute")
            relative = payload.get("relative")
            direction = payload.get("direction")

            threshold_value: float | None = None
            if isinstance(thresholds, Mapping):
                threshold_spec = thresholds.get(metric_name)
                if isinstance(threshold_spec, Mapping):
                    threshold_candidate = threshold_spec.get("relative")
                else:
                    threshold_candidate = threshold_spec
                if threshold_candidate is not None:
                    try:
                        threshold_value = float(threshold_candidate)
                    except (TypeError, ValueError):
                        threshold_value = None

            def _safe_float(value: object) -> float:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return float("nan")

            rows.append(
                {
                    "baseline": baseline_label,
                    "variant": variant_name,
                    "metric": metric_name,
                    "direction": str(direction) if direction is not None else "",
                    "absolute_delta": _safe_float(absolute),
                    "relative_delta": _safe_float(relative),
                    "relative_threshold": _safe_float(threshold_value),
                }
            )

    if not rows:
        return

    outputs["variants"] = _write_csv(
        destination / "variants.csv",
        [
            "baseline",
            "variant",
            "metric",
            "direction",
            "absolute_delta",
            "relative_delta",
            "relative_threshold",
        ],
        rows,
    )


def collate_run_group(run_root: Path, *, aggregates_root: Path | None = None) -> dict[str, Path]:
    """Collate per-seed artifacts under ``run_root`` into aggregate CSV outputs."""

    seed_dirs = _discover_seed_dirs(run_root)
    per_seed_metrics: list[dict[str, float]] = []
    per_seed_metadata: list[dict[str, object]] = []
    compute_efficiency_rows: list[dict[str, object]] = []
    total_wall_time_hours = 0.0
    total_improvement = 0.0
    wall_time_count = 0
    improvement_count = 0
    gpu_util_samples: list[float] = []
    gpu_memory_samples: list[float] = []
    baseline_reference = float("nan")

    for seed_dir in seed_dirs:
        metrics = _load_json(seed_dir / "metrics.json")
        if metrics:
            per_seed_metrics.append({str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        metadata = _load_json(seed_dir / "metadata.json")
        metadata.setdefault("seed", seed_dir.name.split("-", 1)[-1])
        per_seed_metadata.append(metadata)

        compute_record = _load_compute(seed_dir / "compute.csv")
        baseline_mse = _extract_baseline_mse(metadata)
        final_val_loss = float(metrics.get("final_val_loss", float("nan"))) if metrics else float("nan")
        wall_time_s = compute_record.get("wall_time_s", float("nan"))
        wall_time_hours = wall_time_s / 3600.0 if not math.isnan(wall_time_s) else float("nan")
        improvement = (
            baseline_mse - final_val_loss
            if not math.isnan(baseline_mse) and not math.isnan(final_val_loss)
            else float("nan")
        )
        improvement_per_hour = (
            improvement / wall_time_hours
            if not math.isnan(improvement) and not math.isnan(wall_time_hours) and wall_time_hours > 0.0
            else float("nan")
        )

        compute_row = {
            "seed": seed_dir.name,
            "baseline_val_mse": baseline_mse,
            "final_val_mse": final_val_loss,
            "wall_time_s": wall_time_s,
            "wall_time_hours": wall_time_hours,
            "gpu_utilization_mean": compute_record.get("gpu_utilization_mean", float("nan")),
            "gpu_memory_mb_peak": compute_record.get("gpu_memory_mb_peak", float("nan")),
            "improvement": improvement,
            "improvement_per_hour": improvement_per_hour,
        }
        compute_efficiency_rows.append(compute_row)

        if not math.isnan(wall_time_hours):
            total_wall_time_hours += wall_time_hours
            wall_time_count += 1
        if not math.isnan(improvement):
            total_improvement += improvement
            improvement_count += 1
        gpu_util = compute_record.get("gpu_utilization_mean")
        if gpu_util is not None and not math.isnan(gpu_util):
            gpu_util_samples.append(float(gpu_util))
        gpu_mem = compute_record.get("gpu_memory_mb_peak")
        if gpu_mem is not None and not math.isnan(gpu_mem):
            gpu_memory_samples.append(float(gpu_mem))
        if math.isnan(baseline_reference) and not math.isnan(baseline_mse):
            baseline_reference = baseline_mse

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

    if compute_efficiency_rows:
        final_mean = aggregated.get("final_val_loss", {}).get("mean", float("nan"))
        overall_row = {
            "seed": "overall",
            "baseline_val_mse": baseline_reference,
            "final_val_mse": final_mean,
            "wall_time_s": total_wall_time_hours * 3600.0 if wall_time_count > 0 else float("nan"),
            "wall_time_hours": total_wall_time_hours if wall_time_count > 0 else float("nan"),
            "gpu_utilization_mean": (
                sum(gpu_util_samples) / len(gpu_util_samples)
                if gpu_util_samples
                else float("nan")
            ),
            "gpu_memory_mb_peak": max(gpu_memory_samples) if gpu_memory_samples else float("nan"),
            "improvement": total_improvement if improvement_count > 0 else float("nan"),
            "improvement_per_hour": (
                total_improvement / total_wall_time_hours
                if wall_time_count > 0 and improvement_count > 0 and not math.isnan(total_improvement)
                else float("nan")
            ),
        }
        compute_efficiency_rows.append(overall_row)

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

    outputs["compute_efficiency"] = _write_csv(
        destination / "compute.csv",
        [
            "seed",
            "baseline_val_mse",
            "final_val_mse",
            "wall_time_s",
            "wall_time_hours",
            "gpu_utilization_mean",
            "gpu_memory_mb_peak",
            "improvement",
            "improvement_per_hour",
        ],
        compute_efficiency_rows,
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

    _maybe_emit_variant_summary(run_root, destination, outputs)

    artifacts_root = _infer_artifacts_root(run_root)
    if artifacts_root is not None:
        try:
            run_roots = discover_run_roots(artifacts_root / "runs")
            report_path = artifacts_root.parent / "paper_outputs" / "report.html"
            _render_compute_report(run_roots, report_path)
        except Exception:  # pragma: no cover - reporting must not block aggregation
            LOGGER.exception("Failed to render compute report for %s", run_root)

    return outputs


def _render_compute_report(run_roots: Sequence[Path], output_path: Path) -> Path:
    entries: list[dict[str, object]] = []
    for run_root in run_roots:
        destination = _resolve_destination(run_root, None)
        compute_path = destination / "compute.csv"
        if not compute_path.exists():
            continue
        with compute_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        overall = next((row for row in rows if row.get("seed") == "overall"), None)
        if overall is None:
            continue

        metadata = _load_json(run_root / "metadata.json")
        summary_path = run_root / "summary.csv"
        summary_rows = (
            list(csv.DictReader(summary_path.open())) if summary_path.exists() else []
        )
        final_row = next((row for row in summary_rows if row.get("metric") == "final_val_loss"), None)
        final_mean = float(final_row["mean"]) if final_row and final_row.get("mean") else float("nan")

        def _safe_float(value: object) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")

        artifacts_root = _infer_artifacts_root(run_root)
        if artifacts_root is not None:
            try:
                relative = run_root.relative_to(artifacts_root / "runs")
                label = "/".join(relative.parts)
            except ValueError:
                label = run_root.name
        else:
            label = run_root.name

        entries.append(
            {
                "label": label,
                "baseline": _safe_float(overall.get("baseline_val_mse")),
                "final": final_mean,
                "wall_time_hours": _safe_float(overall.get("wall_time_hours")),
                "improvement": _safe_float(overall.get("improvement")),
                "improvement_per_hour": _safe_float(overall.get("improvement_per_hour")),
                "gpu_util": _safe_float(overall.get("gpu_utilization_mean")),
                "gpu_memory": _safe_float(overall.get("gpu_memory_mb_peak")),
                "runs": metadata.get("runs", len(rows) - 1),
            }
        )

    entries.sort(key=lambda item: item.get("improvement_per_hour", float("nan")), reverse=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    rows_html = []
    for record in entries:
        rows_html.append(
            "<tr>"
            f"<td>{record['label']}</td>"
            f"<td>{record['runs']}</td>"
            f"<td>{record['baseline']:.6f}</td>"
            f"<td>{record['final']:.6f}</td>"
            f"<td>{record['wall_time_hours']:.3f}</td>"
            f"<td>{record['gpu_util']:.2f}</td>"
            f"<td>{record['gpu_memory']:.2f}</td>"
            f"<td>{record['improvement']:.6f}</td>"
            f"<td>{record['improvement_per_hour']:.6f}</td>"
            "</tr>"
        )

    if not rows_html:
        rows_html.append("<tr><td colspan=9>No compute data available</td></tr>")

    summary_lines = []
    for record in entries[:3]:
        summary_lines.append(
            "<li>"
            f"{record['label']}: ΔMSE {record['improvement']:.6f} over {record['wall_time_hours']:.3f} h"
            f" (Δ per hour: {record['improvement_per_hour']:.6f})"
            "</li>"
        )

    summary_html = (
        "<ul>" + "".join(summary_lines) + "</ul>" if summary_lines else "<p>No runs summarised.</p>"
    )

    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Compute Efficiency Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 1.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: right; }}
    th {{ background: #f4f4f4; }}
    td:first-child, th:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Compute Efficiency Report</h1>
  <p>Generated at {timestamp}</p>
  <h2>Top Improvements per Compute Hour</h2>
  {summary_html}
  <table>
    <thead>
      <tr>
        <th>Run</th>
        <th>Seeds</th>
        <th>Baseline MSE</th>
        <th>Final MSE (mean)</th>
        <th>Wall Time (h)</th>
        <th>GPU Util. (%)</th>
        <th>GPU Mem Peak (MB)</th>
        <th>ΔMSE</th>
        <th>ΔMSE / h</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
    return output_path


__all__ = ["collate_run_group", "discover_run_roots"]
