"""Multi-run experiment orchestration utilities."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without torch
    torch = None  # type: ignore[assignment]


def _set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on CI environment
        torch.cuda.manual_seed_all(seed)


def _ensure_serialisable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _ensure_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_ensure_serialisable(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def _metric_mean(aggregate: Mapping[str, Mapping[str, Any]], metric: str) -> float:
    entry = aggregate.get(metric)
    if not isinstance(entry, Mapping):
        raise KeyError(f"Metric '{metric}' missing from aggregate summary")
    value = entry.get("mean")
    if value is None:
        raise KeyError(f"Metric '{metric}' does not provide a mean value")
    return float(value)


def _normalise_thresholds(
    thresholds: Mapping[str, object] | None,
) -> Dict[str, Dict[str, float | str | None]]:
    normalised: Dict[str, Dict[str, float | str | None]] = {}
    if not thresholds:
        return normalised

    for metric, spec in thresholds.items():
        direction = "decrease"
        relative: float | None = None
        absolute: float | None = None
        if isinstance(spec, Mapping):
            direction = str(spec.get("direction", "decrease")).lower()
            rel_value = spec.get("relative")
            if rel_value is not None:
                relative = float(rel_value)
            abs_value = spec.get("absolute")
            if abs_value is not None:
                absolute = float(abs_value)
        else:
            relative = float(spec)
        if direction not in {"decrease", "increase"}:
            raise ValueError(f"Unsupported direction '{direction}' for metric '{metric}'")
        normalised[metric] = {"direction": direction, "relative": relative, "absolute": absolute}
    return normalised


def _compute_improvement(
    baseline_mean: float,
    variant_mean: float,
    *,
    direction: str,
) -> Tuple[float, float]:
    if direction == "increase":
        delta = variant_mean - baseline_mean
    else:
        delta = baseline_mean - variant_mean

    if math.isclose(baseline_mean, 0.0, abs_tol=1e-12):
        relative = float("nan")
    else:
        relative = delta / abs(baseline_mean)
    return delta, relative


@dataclass
class RunRecord:
    """Represents a single seeded run."""

    seed: int
    metrics: Mapping[str, float]
    extra: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass
class MultiRunResult:
    """Aggregated output of a multi-run experiment."""

    runs: List[RunRecord]
    aggregate: Dict[str, Dict[str, float]]
    metadata_path: Path
    seeds: List[int]


@dataclass(frozen=True)
class VariantRunCollection:
    """Container for a set of variant runs executed with shared seeds."""

    baseline: str
    results: Dict[str, MultiRunResult]
    summary_path: Path


class MultiRunExperiment:
    """Execute multiple seeded runs for a configuration and aggregate metrics."""

    def __init__(
        self,
        run_fn: Callable[[MutableMapping[str, Any]], Mapping[str, Any]],
        num_runs: int = 5,
        base_seed: int = 0,
    ) -> None:
        if num_runs < 5:
            raise ValueError("At least five runs are required for robust statistics")
        self.run_fn = run_fn
        self.num_runs = num_runs
        self.base_seed = base_seed

    def _aggregate_metrics(self, records: Iterable[RunRecord]) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, List[float]] = {}
        for record in records:
            for key, value in record.metrics.items():
                metrics.setdefault(key, []).append(float(value))

        summary: Dict[str, Dict[str, float]] = {}
        for name, values in metrics.items():
            arr = np.asarray(values, dtype=np.float64)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            ci_radius = 1.96 * std / math.sqrt(arr.size) if arr.size > 1 else 0.0
            summary[name] = {
                "mean": mean,
                "std": std,
                "ci95_low": mean - ci_radius,
                "ci95_high": mean + ci_radius,
                "n": float(arr.size),
            }
        return summary

    def run(
        self,
        config: MutableMapping[str, Any],
        *,
        run_id: str,
        output_dir: Path | str = Path("artifacts"),
        seeds: Optional[Sequence[int]] = None,
    ) -> MultiRunResult:
        seeds = list(seeds or range(self.base_seed, self.base_seed + self.num_runs))
        if len(seeds) < self.num_runs:
            raise ValueError("Insufficient seeds for requested runs")

        output_dir = Path(output_dir)
        run_root = output_dir / run_id
        run_root.mkdir(parents=True, exist_ok=True)

        records: List[RunRecord] = []
        metadata_runs: List[Dict[str, Any]] = []

        for run_index, seed in enumerate(seeds[: self.num_runs]):
            _set_random_seeds(seed)
            config_copy = dict(config)
            config_copy.update({"seed": seed, "run_index": run_index})
            result = self.run_fn(config_copy)
            if "metrics" in result:
                metrics = result["metrics"]
                extras = {k: v for k, v in result.items() if k != "metrics"}
            else:
                metrics = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
                extras = {k: v for k, v in result.items() if k not in metrics}

            record = RunRecord(seed=seed, metrics=metrics, extra=extras)
            records.append(record)
            metadata_runs.append(
                {
                    "seed": seed,
                    "run_index": run_index,
                    "metrics": _ensure_serialisable(metrics),
                    "extra": _ensure_serialisable(extras),
                }
            )

        aggregate = self._aggregate_metrics(records)

        metadata = {
            "config": _ensure_serialisable(config),
            "seeds": seeds[: self.num_runs],
            "runs": metadata_runs,
            "aggregate": _ensure_serialisable(aggregate),
        }

        metadata_path = run_root / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        return MultiRunResult(
            runs=records,
            aggregate=aggregate,
            metadata_path=metadata_path,
            seeds=seeds[: self.num_runs],
        )

    def run_variants(
        self,
        config: MutableMapping[str, Any],
        *,
        run_id: str,
        variants: Mapping[str, Mapping[str, Any]],
        baseline: str,
        output_dir: Path | str = Path("artifacts"),
        seeds: Optional[Sequence[int]] = None,
        improvement_thresholds: Mapping[str, object] | None = None,
    ) -> VariantRunCollection:
        """Execute multiple configuration variants sharing the same seed schedule.

        Parameters
        ----------
        config:
            Base configuration shared across all variants. Each variant may override
            specific keys but the remaining parameters (e.g., training budgets) are
            preserved to ensure apples-to-apples comparisons.
        run_id:
            Identifier for the family of runs. Individual variants are written under
            ``<output_dir>/<run_id>/<variant_name>``.
        variants:
            Mapping from variant name to configuration overrides. Overrides are merged
            on top of ``config``.
        baseline:
            Name of the variant representing the baseline configuration. Improvements
            for other variants are measured relative to this entry.
        output_dir:
            Directory where artefacts should be persisted.
        seeds:
            Explicit seed schedule to reuse for every variant. When omitted the
            experiment's default sequential seeds are used.
        improvement_thresholds:
            Optional mapping specifying minimum acceptable improvements. Values may be
            floats (interpreted as relative improvements for lower-is-better metrics)
            or dictionaries containing ``direction`` (``"decrease"`` or
            ``"increase"``) alongside ``relative``/``absolute`` thresholds.

        Returns
        -------
        VariantRunCollection
            Object containing the per-variant results and the path to the summary
            manifest describing deltas.
        """

        if baseline not in variants:
            raise KeyError(f"Baseline variant '{baseline}' not provided in variants mapping")

        seeds_list = list(seeds or range(self.base_seed, self.base_seed + self.num_runs))
        if len(seeds_list) < self.num_runs:
            raise ValueError("Insufficient seeds supplied for variant execution")

        thresholds = _normalise_thresholds(improvement_thresholds)

        output_dir = Path(output_dir)
        variant_root = output_dir / run_id
        variant_root.mkdir(parents=True, exist_ok=True)

        base_config_serialisable = _ensure_serialisable(config)

        results: Dict[str, MultiRunResult] = {}
        summary_variants: List[Dict[str, Any]] = []

        baseline_result: MultiRunResult | None = None

        for variant_name, overrides in variants.items():
            merged_config: Dict[str, Any] = dict(config)
            merged_config.update(dict(overrides))

            result = self.run(
                merged_config,
                run_id=f"{run_id}/{variant_name}",
                output_dir=output_dir,
                seeds=seeds_list,
            )

            results[variant_name] = result
            if variant_name == baseline:
                baseline_result = result

            summary_variants.append(
                {
                    "name": variant_name,
                    "overrides": _ensure_serialisable(overrides),
                    "config": _ensure_serialisable(merged_config),
                    "aggregate": _ensure_serialisable(result.aggregate),
                    "metadata_path": str(result.metadata_path),
                    "seeds": list(result.seeds),
                }
            )

        if baseline_result is None:
            raise KeyError(f"Baseline variant '{baseline}' did not execute successfully")

        expected_seeds = list(baseline_result.seeds)
        for variant_name, result in results.items():
            if list(result.seeds) != expected_seeds:
                raise ValueError(
                    "Variant runs must reuse the same seed schedule;"
                    f" expected {expected_seeds} but variant '{variant_name}' reported {result.seeds}"
                )

        baseline_aggregate = baseline_result.aggregate

        for entry in summary_variants:
            name = entry["name"]
            if name == baseline:
                entry["deltas"] = {}
                continue

            variant_result = results[name]
            deltas: Dict[str, Dict[str, float | str]] = {}

            for metric, spec in thresholds.items():
                base_mean = _metric_mean(baseline_aggregate, metric)
                variant_mean = _metric_mean(variant_result.aggregate, metric)
                delta, relative = _compute_improvement(
                    base_mean, variant_mean, direction=str(spec["direction"])
                )
                deltas[metric] = {
                    "direction": spec["direction"],
                    "absolute": delta,
                    "relative": relative,
                }

                rel_threshold = spec.get("relative")
                if rel_threshold is not None:
                    if math.isnan(relative) or relative < float(rel_threshold) - 1e-12:
                        raise ValueError(
                            f"Variant '{name}' underperforms baseline for metric '{metric}':"
                            f" relative improvement {relative:.6f} < {rel_threshold}"
                        )

                abs_threshold = spec.get("absolute")
                if abs_threshold is not None and delta < float(abs_threshold) - 1e-12:
                    raise ValueError(
                        f"Variant '{name}' underperforms baseline for metric '{metric}':"
                        f" absolute improvement {delta:.6f} < {abs_threshold}"
                    )

            entry["deltas"] = deltas

        summary_payload = {
            "baseline": baseline,
            "config": base_config_serialisable,
            "seeds": expected_seeds,
            "thresholds": _ensure_serialisable(thresholds),
            "variants": summary_variants,
        }

        summary_path = variant_root / "variants.json"
        summary_path.write_text(json.dumps(summary_payload, indent=2))

        return VariantRunCollection(baseline=baseline, results=results, summary_path=summary_path)

