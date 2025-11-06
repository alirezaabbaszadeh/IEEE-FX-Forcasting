"""Hyperparameter optimisation harness using Optuna when available."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping

import pandas as pd

from .runner import MultiRunExperiment
from src.analysis.hparam import plot_response_curves

try:  # pragma: no cover - optional dependency probing
    import scipy.stats.qmc  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - SciPy not installed
    _SCIPY_AVAILABLE = False
else:  # pragma: no cover - SciPy available
    _SCIPY_AVAILABLE = True

try:  # pragma: no cover - exercised indirectly when Optuna is installed
    import optuna  # type: ignore
except ImportError:  # pragma: no cover - fallback path used in tests
    from . import _optuna_stub as optuna  # type: ignore


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


class HyperparameterSearch:
    """Optuna-backed search wrapper using deterministic multi-run evaluations."""

    def __init__(
        self,
        run_fn: MultiRunExperiment | Callable[[MutableMapping[str, Any]], Mapping[str, Any]],
        base_config: Mapping[str, Any],
        search_space: Mapping[str, Mapping[str, Any]],
        metric: str,
        run_id: str,
        output_dir: Path | str = Path("artifacts"),
        direction: str = "maximize",
        *,
        num_runs: int = 5,
        base_seed: int = 0,
        top_k: int = 5,
        governance: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(run_fn, MultiRunExperiment):
            self.runner = run_fn
        else:
            self.runner = MultiRunExperiment(run_fn, num_runs=num_runs, base_seed=base_seed)
        self.base_config = dict(base_config)
        self.search_space = search_space
        self.metric = metric
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.direction = direction
        self.top_k = max(1, int(top_k))
        self._trial_records: list[Dict[str, Any]] = []
        self._artifact_root = self.output_dir / "hparam" / self.run_id
        self._trials_dir = self._artifact_root / "trials"
        self._multirun_root = self._artifact_root / "multirun"
        self._governance = dict(governance or {})

    @staticmethod
    def _hash_config(config: Mapping[str, Any]) -> str:
        payload = json.dumps(config, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def _resolve_model_name(self) -> str | None:
        model_cfg = self.base_config.get("model")
        if isinstance(model_cfg, Mapping):
            name = model_cfg.get("name")
            if isinstance(name, str):
                return name.lower()
        model_name = self.base_config.get("model_name")
        if isinstance(model_name, str):
            return model_name.lower()
        return None

    def _lookup_trial_limit(self) -> tuple[str, int | None]:
        model_name = self._resolve_model_name() or "model"

        def _extract(source: Mapping[str, Any] | None) -> int | None:
            if not source:
                return None
            mapping: Mapping[str, Any] | Any = source
            if isinstance(mapping, Mapping) and "max_hpo_trials" in mapping:
                mapping = mapping["max_hpo_trials"]
            if isinstance(mapping, Mapping):
                key = model_name.lower()
                if key in mapping and mapping[key] is not None:
                    return int(mapping[key])
                for fallback in ("_default", "default"):
                    if fallback in mapping and mapping[fallback] is not None:
                        return int(mapping[fallback])
            elif isinstance(mapping, (int, float)):
                return int(mapping)
            return None

        limit = _extract(self._governance)
        if limit is None:
            governance_cfg = self.base_config.get("governance")
            if isinstance(governance_cfg, Mapping):
                limit = _extract(governance_cfg)
        return model_name, limit

    def _suggest_parameters(self, trial) -> Dict[str, Any]:
        config = copy.deepcopy(self.base_config)
        for name, spec in self.search_space.items():
            param_type = spec.get("type")
            if param_type == "float":
                config[name] = trial.suggest_float(
                    name,
                    float(spec["low"]),
                    float(spec["high"]),
                    log=bool(spec.get("log", False)),
                )
            elif param_type == "int":
                config[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
            elif param_type == "categorical":
                config[name] = trial.suggest_categorical(name, spec["choices"])
            else:  # pragma: no cover - defensive path
                raise ValueError(f"Unsupported parameter type: {param_type}")
        return config

    def _trial_objective(self, trial) -> float:
        config = self._suggest_parameters(trial)
        trial_run_id = f"{self.run_id}_trial{trial.number}"
        result = self.runner.run(config, run_id=trial_run_id, output_dir=self._multirun_root)
        metric_summary = result.aggregate.get(self.metric)
        if not metric_summary:
            raise KeyError(f"Metric '{self.metric}' not reported by run function")
        config_serialisable = _ensure_serialisable(config)
        aggregate_serialisable = _ensure_serialisable(result.aggregate)
        seeds = list(result.seeds)
        config_hash = self._hash_config(config_serialisable)

        record = {
            "trial": trial.number,
            "config": config_serialisable,
            "params": {name: config_serialisable.get(name) for name in self.search_space},
            "aggregate": aggregate_serialisable,
            "seeds": seeds,
            "config_hash": config_hash,
            "metadata_path": str(result.metadata_path),
            "value": float(metric_summary["mean"]),
        }
        self._trial_records.append(record)

        trial_payload = {
            "trial": trial.number,
            "config_hash": config_hash,
            "config": record["config"],
            "params": record["params"],
            "aggregate": record["aggregate"],
            "seeds": seeds,
            "metric": self.metric,
            "objective": record["value"],
            "metadata_path": record["metadata_path"],
        }
        trial_path = self._trials_dir / f"{trial.number:04d}_{config_hash}.json"
        trial_path.write_text(json.dumps(trial_payload, indent=2))

        trial.set_user_attr("config_hash", config_hash)
        trial.set_user_attr("aggregate", result.aggregate)
        trial.set_user_attr("seeds", seeds)
        trial.set_user_attr("metadata_path", str(result.metadata_path))

        return record["value"]

    def optimize(self, n_trials: int, sampler: str = "sobol") -> Dict[str, Any]:
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        self._trials_dir.mkdir(parents=True, exist_ok=True)
        self._multirun_root.mkdir(parents=True, exist_ok=True)
        self._trial_records = []

        model_name, trial_limit = self._lookup_trial_limit()
        if trial_limit is not None and n_trials > int(trial_limit):
            raise ValueError(
                f"Requested {n_trials} trials for model '{model_name}' exceeds governance limit {trial_limit}"
            )

        if sampler.lower() == "sobol":
            if hasattr(optuna.samplers, "SobolSampler"):
                sampler_impl = optuna.samplers.SobolSampler()
            else:
                if _SCIPY_AVAILABLE:
                    sampler_impl = optuna.samplers.QMCSampler(qmc_type="sobol")
                else:
                    sampler_impl = optuna.samplers.RandomSampler()
        elif sampler.lower() in {"tpe", "bayesian"}:
            sampler_impl = optuna.samplers.TPESampler()
        else:  # pragma: no cover - defensive path
            raise ValueError(f"Unknown sampler: {sampler}")

        study = optuna.create_study(direction=self.direction, sampler=sampler_impl)
        study.optimize(self._trial_objective, n_trials=n_trials)

        sort_reverse = self.direction.lower().startswith("max")
        flattened_records: list[Dict[str, Any]] = []
        for record in self._trial_records:
            row: Dict[str, Any] = {
                "trial": record["trial"],
                "value": record["value"],
                "config_hash": record["config_hash"],
            }
            row.update(record["params"])
            for metric_name, stats in record["aggregate"].items():
                for stat_name, stat_value in stats.items():
                    row[f"{metric_name}_{stat_name}"] = float(stat_value)
            flattened_records.append(row)

        results_df = pd.DataFrame(flattened_records)
        results_path = self._artifact_root / "search_results.csv"
        results_df.to_csv(results_path, index=False)

        top_trials = sorted(self._trial_records, key=lambda item: item["value"], reverse=sort_reverse)
        top_trials = top_trials[: min(self.top_k, len(top_trials))]
        top_payload = []
        for rank, record in enumerate(top_trials, start=1):
            top_payload.append(
                {
                    "rank": rank,
                    "trial": record["trial"],
                    "config_hash": record["config_hash"],
                    "value": record["value"],
                    "metric": self.metric,
                    "aggregate": record["aggregate"],
                    "params": record["params"],
                    "config": record["config"],
                    "seeds": record["seeds"],
                    "metadata_path": record["metadata_path"],
                }
            )

        top_trials_path = self._artifact_root / "top_trials.json"
        top_trials_path.write_text(json.dumps(_ensure_serialisable(top_payload), indent=2))

        metric_column = f"{self.metric}_mean"
        if metric_column not in results_df.columns:
            metric_column = "value"

        artefacts = plot_response_curves(
            results_df,
            metric_column=metric_column,
            output_dir=self._artifact_root / "plots",
            parameter_columns=list(self.search_space.keys()),
        )
        plot_paths = [paths["plot"] for paths in artefacts.values()]
        summary_paths = [paths["summary"] for paths in artefacts.values()]

        return {
            "study": study,
            "best_params": getattr(study.best_trial, "params", {}),
            "best_value": getattr(study, "best_value", None),
            "results_path": results_path,
            "plot_paths": plot_paths,
            "summary_paths": summary_paths,
            "top_trials_path": top_trials_path,
            "records": copy.deepcopy(self._trial_records),
        }

