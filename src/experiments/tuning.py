"""Hyperparameter tuning utilities built on top of Optuna and multi-run evaluation."""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from src.analysis.hparam import plot_response_curves
from src.models.forecasting import ModelConfig
from src.training.engine import TrainerConfig

from .runner import MultiRunExperiment

LOGGER = logging.getLogger(__name__)


try:  # pragma: no cover - exercised indirectly when Optuna is installed
    import optuna  # type: ignore
except ImportError:  # pragma: no cover - fallback path used in tests
    from . import _optuna_stub as optuna  # type: ignore


@dataclass
class SearchSpaceConfig:
    """Container describing the hyper-parameter search configuration."""

    model: Mapping[str, Mapping[str, Any]]
    training: Mapping[str, Mapping[str, Any]]
    metric: str
    direction: str
    seeds: Sequence[int]
    top_k: int = 5
    sampler_seed: int | None = None


def _normalise_windows(
    windows: Mapping[str, Any] | Sequence[Tuple[str, Any]]
) -> List[Tuple[str, Any]]:
    if isinstance(windows, Mapping):
        return [(str(name), payload) for name, payload in windows.items()]
    return [(str(name), payload) for name, payload in windows]


def load_search_space(path: Path | str) -> SearchSpaceConfig:
    """Load a search space definition from YAML."""

    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        raise ValueError("Search space YAML cannot be empty")

    model_space = raw.get("model", {})
    training_space = raw.get("training", {})
    evaluation = raw.get("evaluation", {})

    metric = evaluation.get("metric", "val_loss")
    direction = evaluation.get("direction", "minimize")
    seeds = evaluation.get("seeds", list(range(5)))
    top_k = int(evaluation.get("top_k", 5))
    sampler_seed = evaluation.get("sampler_seed")

    if not isinstance(seeds, Sequence) or not seeds:
        raise ValueError("Evaluation seeds must be a non-empty sequence")

    if len(seeds) < 5:
        raise ValueError("At least five seeds are required for multi-run evaluation")

    return SearchSpaceConfig(
        model=model_space,
        training=training_space,
        metric=str(metric),
        direction=str(direction),
        seeds=[int(seed) for seed in seeds],
        top_k=top_k,
        sampler_seed=None if sampler_seed is None else int(sampler_seed),
    )


EvaluationFn = Callable[[ModelConfig, TrainerConfig, int, Any], Mapping[str, float]]


class HyperparameterTuner:
    """Run Sobol/Optuna search over model and trainer spaces using multi-run evaluation."""

    def __init__(
        self,
        evaluation_fn: EvaluationFn,
        *,
        base_model_cfg: ModelConfig,
        base_trainer_cfg: TrainerConfig,
        windows: Mapping[str, Any] | Sequence[Tuple[str, Any]],
        search_space: SearchSpaceConfig,
        run_id: str,
        output_dir: Path | str = Path("artifacts") / "hparam",
        evaluation_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not windows:
            raise ValueError("At least one evaluation window must be supplied")

        self.evaluation_fn = evaluation_fn
        self.base_model_cfg = base_model_cfg
        self.base_trainer_cfg = base_trainer_cfg
        self.windows = _normalise_windows(windows)
        self.search_space = search_space
        self.metric = search_space.metric
        self.direction = search_space.direction
        self.seeds = list(search_space.seeds)
        self.top_k = search_space.top_k
        self._evaluation_kwargs = dict(evaluation_kwargs or {})
        self.run_id = run_id
        self.artifact_root = Path(output_dir) / run_id
        self.artifact_root.mkdir(parents=True, exist_ok=True)

        self._trial_summaries: list[dict[str, Any]] = []
        self._window_records: list[dict[str, Any]] = []
        self._seed_records: list[dict[str, Any]] = []
        self._parameter_columns = [
            *(f"model.{name}" for name in search_space.model.keys()),
            *(f"training.{name}" for name in search_space.training.keys()),
        ]

    @property
    def evaluation_seeds(self) -> tuple[int, ...]:
        """Return the deterministic seed schedule used for evaluation."""

        return tuple(self.seeds)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Path | str,
        evaluation_fn: EvaluationFn,
        *,
        base_model_cfg: ModelConfig,
        base_trainer_cfg: TrainerConfig,
        windows: Mapping[str, Any] | Sequence[Tuple[str, Any]],
        run_id: str,
        output_dir: Path | str = Path("artifacts") / "hparam",
        evaluation_kwargs: Mapping[str, Any] | None = None,
    ) -> "HyperparameterTuner":
        """Construct a tuner from a YAML search definition."""

        search_space = load_search_space(yaml_path)
        return cls(
            evaluation_fn,
            base_model_cfg=base_model_cfg,
            base_trainer_cfg=base_trainer_cfg,
            windows=windows,
            search_space=search_space,
            run_id=run_id,
            output_dir=output_dir,
            evaluation_kwargs=evaluation_kwargs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def optimize(self, n_trials: int) -> Dict[str, Any]:
        """Launch the Sobol search and persist artefacts."""

        sampler = self._create_sampler()
        study = optuna.create_study(direction=self.direction, sampler=sampler)
        study.optimize(lambda trial: self._objective(trial), n_trials=n_trials)

        results_df = self._build_results_dataframe()
        results_path = self.artifact_root / "search_results.csv"
        results_df.to_csv(results_path, index=False)

        ranking_df = self._build_ranking_dataframe()
        ranking_path = self.artifact_root / "ranking_stability.csv"
        ranking_df.to_csv(ranking_path, index=False)

        plots_dir = self.artifact_root / "plots"
        plot_paths = plot_response_curves(
            results_df,
            metric_column="value",
            output_dir=plots_dir,
            parameter_columns=self._parameter_columns,
        )

        self._log_top_configs()

        return {
            "study": study,
            "best_params": getattr(study.best_trial, "params", {}),
            "best_value": getattr(study, "best_value", None),
            "results_path": results_path,
            "ranking_path": ranking_path,
            "plot_paths": plot_paths,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_sampler(self):
        seed = self.search_space.sampler_seed
        sobol = getattr(optuna.samplers, "SobolSampler", None)
        if sobol is not None:
            return sobol(seed=seed)
        return optuna.samplers.QMCSampler(qmc_type="sobol", seed=seed)

    def _suggest_block(
        self, trial: Any, block_name: str, space: Mapping[str, Mapping[str, Any]]
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name, spec in space.items():
            param_name = f"{block_name}.{name}"
            param_type = spec.get("type", "float")
            if param_type == "float":
                params[name] = trial.suggest_float(
                    param_name,
                    float(spec["low"]),
                    float(spec["high"]),
                    log=bool(spec.get("log", False)),
                )
            elif param_type == "int":
                params[name] = trial.suggest_int(
                    param_name,
                    int(spec["low"]),
                    int(spec["high"]),
                )
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(param_name, list(spec["choices"]))
            else:  # pragma: no cover - defensive guard
                raise ValueError(f"Unsupported parameter type: {param_type}")
        return params

    def _objective(self, trial: Any) -> float:
        model_params = self._suggest_block(trial, "model", self.search_space.model)
        trainer_params = self._suggest_block(trial, "training", self.search_space.training)

        model_cfg = replace(self.base_model_cfg, **model_params)
        trainer_cfg = replace(self.base_trainer_cfg, **trainer_params)

        window_summaries: list[dict[str, Any]] = []
        window_means: list[float] = []

        for window_label, window_payload in self.windows:
            run_config: MutableMapping[str, Any] = {
                "model": dataclasses.asdict(model_cfg),
                "trainer": dataclasses.asdict(trainer_cfg),
                "window": window_label,
            }

            def _run_once(config: MutableMapping[str, Any]) -> Mapping[str, Any]:
                seed = int(config["seed"])
                metrics = self.evaluation_fn(
                    model_cfg,
                    trainer_cfg,
                    seed,
                    window_payload,
                    **self._evaluation_kwargs,
                )
                return {"metrics": dict(metrics)}

            experiment = MultiRunExperiment(
                _run_once,
                num_runs=len(self.seeds),
                base_seed=min(self.seeds),
            )
            result = experiment.run(
                run_config,
                run_id=f"trial_{trial.number}/window_{window_label}",
                output_dir=self.artifact_root,
                seeds=self.seeds,
            )

            metric_summary = result.aggregate.get(self.metric)
            if not metric_summary:
                raise KeyError(f"Metric '{self.metric}' not reported by evaluation function")

            window_means.append(float(metric_summary["mean"]))
            summary_record = {
                "window": window_label,
                "mean": float(metric_summary["mean"]),
                "std": float(metric_summary.get("std", float("nan"))),
                "ci95_low": float(metric_summary.get("ci95_low", float("nan"))),
                "ci95_high": float(metric_summary.get("ci95_high", float("nan"))),
                "n": float(metric_summary.get("n", 0.0)),
            }
            window_summaries.append(summary_record)
            window_record = dict(summary_record)
            window_record["trial"] = trial.number
            self._window_records.append(window_record)

            for run in result.runs:
                metric_value = run.metrics.get(self.metric)
                if metric_value is None:
                    continue
                self._seed_records.append(
                    {
                        "trial": trial.number,
                        "window": window_label,
                        "seed": run.seed,
                        "metric": float(metric_value),
                    }
                )

        objective_value = float(np.mean(window_means)) if window_means else float("nan")

        trial_summary = {
            "trial": trial.number,
            "value": objective_value,
            "model_params": model_params,
            "trainer_params": trainer_params,
            "windows": window_summaries,
        }
        self._trial_summaries.append(trial_summary)

        return objective_value

    def _build_results_dataframe(self) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for summary in self._trial_summaries:
            row = {"trial": summary["trial"], "value": summary["value"]}
            for name, value in summary["model_params"].items():
                row[f"model.{name}"] = value
            for name, value in summary["trainer_params"].items():
                row[f"training.{name}"] = value
            records.append(row)
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("trial").reset_index(drop=True)
        return df

    def _build_ranking_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._window_records)
        if df.empty:
            return df
        ascending = self.direction != "maximize"
        df["rank"] = (
            df.groupby("window")["mean"].rank(method="min", ascending=ascending)
        )
        df = df.sort_values(["window", "rank", "trial"]).reset_index(drop=True)
        return df

    def _log_top_configs(self) -> None:
        if not self._trial_summaries:
            return

        reverse = self.direction == "maximize"
        sorted_trials = sorted(
            self._trial_summaries,
            key=lambda item: item["value"],
            reverse=reverse,
        )
        LOGGER.info("Top %d configurations for run %s:", self.top_k, self.run_id)
        for rank, summary in enumerate(sorted_trials[: self.top_k], start=1):
            LOGGER.info("  #%d trial %d -> %.6f", rank, summary["trial"], summary["value"])
            for window in summary["windows"]:
                seed_metrics = [
                    record
                    for record in self._seed_records
                    if record["trial"] == summary["trial"]
                    and record["window"] == window["window"]
                ]
                seed_blob = "; ".join(
                    f"seed={record['seed']}: {record['metric']:.6f}"
                    for record in seed_metrics
                )
                LOGGER.info(
                    "      window %s mean=%.6f std=%.6f | %s",
                    window["window"],
                    window["mean"],
                    window["std"],
                    seed_blob or "no per-seed metrics",
                )


__all__ = ["HyperparameterTuner", "SearchSpaceConfig", "load_search_space"]

