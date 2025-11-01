"""Multi-run experiment orchestration utilities."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch


def _set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
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

        return MultiRunResult(runs=records, aggregate=aggregate, metadata_path=metadata_path)

