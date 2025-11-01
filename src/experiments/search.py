"""Hyperparameter optimisation harness using Optuna when available."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

from .runner import MultiRunExperiment

try:  # pragma: no cover - exercised indirectly when Optuna is installed
    import optuna  # type: ignore
except ImportError:  # pragma: no cover - fallback path used in tests
    from . import _optuna_stub as optuna  # type: ignore


class HyperparameterSearch:
    """Optuna-backed search wrapper operating on top of :class:`MultiRunExperiment`."""

    def __init__(
        self,
        runner: MultiRunExperiment,
        base_config: Mapping[str, Any],
        search_space: Mapping[str, Mapping[str, Any]],
        metric: str,
        run_id: str,
        output_dir: Path | str = Path("artifacts"),
        direction: str = "maximize",
    ) -> None:
        self.runner = runner
        self.base_config = dict(base_config)
        self.search_space = search_space
        self.metric = metric
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.direction = direction

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
        result = self.runner.run(config, run_id=trial_run_id, output_dir=self.output_dir)
        metric_summary = result.aggregate.get(self.metric)
        if not metric_summary:
            raise KeyError(f"Metric '{self.metric}' not reported by run function")
        return float(metric_summary["mean"])

    def optimize(self, n_trials: int, sampler: str = "sobol") -> Dict[str, Any]:
        if sampler.lower() == "sobol":
            if hasattr(optuna.samplers, "SobolSampler"):
                sampler_impl = optuna.samplers.SobolSampler()
            else:
                sampler_impl = optuna.samplers.QMCSampler(qmc_type="sobol")
        elif sampler.lower() in {"tpe", "bayesian"}:
            sampler_impl = optuna.samplers.TPESampler()
        else:  # pragma: no cover - defensive path
            raise ValueError(f"Unknown sampler: {sampler}")

        study = optuna.create_study(direction=self.direction, sampler=sampler_impl)
        study.optimize(self._trial_objective, n_trials=n_trials)

        records = []
        for trial in study.trials:
            row = {"trial": trial.number, "value": getattr(trial, "value", None)}
            row.update(trial.params)
            records.append(row)

        results_df = pd.DataFrame(records)
        output_path = self.output_dir / self.run_id
        output_path.mkdir(parents=True, exist_ok=True)
        results_path = output_path / "search_results.csv"
        results_df.to_csv(results_path, index=False)

        return {
            "study": study,
            "best_params": getattr(study.best_trial, "params", {}),
            "best_value": getattr(study, "best_value", None),
            "results_path": results_path,
        }

