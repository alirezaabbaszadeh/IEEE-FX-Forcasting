"""Core orchestration utilities shared across experiment variants.

This module captures the responsibilities that were historically duplicated in
``v_*/MainClass.py``.  The :class:`TimeSeriesOrchestrator` centralises
run-directory creation, logging bootstrap, dependency wiring, and the
end-to-end training/evaluation flow while remaining agnostic to the concrete
implementations of the DataLoader, ModelBuilder, Trainer, Evaluator, and
HistoryManager.

Each experiment version should inject its own dependency factories (or register
factories with :class:`DependencyFactoryRegistry`) so that bespoke behaviour
lives outside of the shared core.  The orchestrator focuses solely on
coordinating lifecycle steps and recording metadata about the run.
"""

from __future__ import annotations

import datetime
import inspect
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Protocol, Tuple, Union

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    import tensorflow as tf


class TrainerProtocol(Protocol):
    """Protocol describing the minimal trainer surface expected by the orchestrator."""

    def train(self) -> Any:
        """Execute training and return the framework-specific history object."""


@dataclass
class DependencyFactoryRegistry:
    """Registry that stores named factories for experiment-specific dependencies."""

    factories: MutableMapping[str, Callable[..., Any]] = field(default_factory=dict)

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        """Register or overwrite a factory under ``name``."""
        self.factories[name] = factory

    def get(self, name: str) -> Callable[..., Any]:
        """Retrieve a previously registered factory."""
        if name not in self.factories:
            available = ", ".join(sorted(self.factories)) or "<empty>"
            raise KeyError(f"Factory '{name}' is not registered. Available: {available}")
        return self.factories[name]

    def create(self, name: str, /, **kwargs: Any) -> Any:
        """Instantiate a dependency using the stored factory."""
        return self.get(name)(**kwargs)


FactoryReference = Union[str, Callable[..., Any]]


@dataclass
class OrchestratorDependencies:
    """Dependency wiring for the orchestrator.

    Each attribute can be a callable factory or a string reference to a factory
    stored inside an optional :class:`DependencyFactoryRegistry`.
    """

    data_loader: FactoryReference
    model_builder: FactoryReference
    trainer: FactoryReference
    evaluator: FactoryReference
    history_manager: FactoryReference
    registry: Optional[DependencyFactoryRegistry] = None

    def get_factory(self, name: str) -> Callable[..., Any]:
        value = getattr(self, name)
        if callable(value):
            return value
        if isinstance(value, str):
            if self.registry is None:
                raise ValueError(
                    f"Dependency '{name}' requested registry lookup for key '{value}' but no registry was provided."
                )
            return self.registry.get(value)
        raise TypeError(f"Unsupported factory reference for '{name}': {type(value)!r}")


@dataclass
class OrchestratorConfig:
    """Container for configuration sections consumed by the orchestrator."""

    file_path: str
    base_dir: str = "TimeSeries_Project_Runs/"
    data: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    callbacks: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorHooks:
    """Optional lifecycle hooks for experiment-specific extensibility."""

    before_data: Optional[Callable[["TimeSeriesOrchestrator"], None]] = None
    after_data: Optional[Callable[["TimeSeriesOrchestrator", Dict[str, Any]], None]] = None
    before_model_build: Optional[Callable[["TimeSeriesOrchestrator", Dict[str, Any]], None]] = None
    after_model_build: Optional[Callable[["TimeSeriesOrchestrator", Any], None]] = None
    before_training: Optional[Callable[["TimeSeriesOrchestrator", Dict[str, Any]], None]] = None
    after_training: Optional[Callable[["TimeSeriesOrchestrator", Any], None]] = None
    before_evaluation: Optional[Callable[["TimeSeriesOrchestrator"], None]] = None
    after_evaluation: Optional[Callable[["TimeSeriesOrchestrator", Any], None]] = None
    finalize: Optional[Callable[["TimeSeriesOrchestrator", Dict[str, Any]], None]] = None


class TimeSeriesOrchestrator:
    """Shared pipeline coordinator used by experiment variants.

    The orchestrator owns responsibilities that were previously duplicated in
    each ``TimeSeriesModel`` implementation:

    * Create a run directory and wire up run-scoped logging.
    * Instantiate helper components (data loader, model builder, trainer,
      evaluator, history manager).
    * Persist hyperparameters, history artefacts, and model checkpoints.
    * Provide callbacks for trainers that need run-level information
      (``save_model_per_epoch`` or ``record_epoch_duration``).
    * Execute the canonical training/evaluation flow while exposing hook points
      for bespoke behaviour.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        dependencies: OrchestratorDependencies,
        hooks: Optional[OrchestratorHooks] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.dependencies = dependencies
        self.hooks = hooks or OrchestratorHooks()
        self.logger = logger or logging.getLogger(__name__)

        self.model: Optional["tf.keras.Model"] = None
        self.trainer: Optional[TrainerProtocol] = None
        self.evaluator: Optional[Any] = None
        self.history_manager: Optional[Any] = None

        self.epoch_durations: list[float] = []
        self.training_summary: Dict[str, Any] = {}

        self._setup_run_environment()
        self._initialise_history_manager()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def _setup_run_environment(self) -> None:
        os.makedirs(self.config.base_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = os.path.join(self.config.base_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.log_file_path = os.path.join(self.run_dir, "run_pipeline_log.txt")
        self.file_log_handler = logging.FileHandler(self.log_file_path, mode="w")
        self.file_log_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s (%(filename)s:%(lineno)d)"
        )
        self.file_log_handler.setFormatter(formatter)
        logging.getLogger().addHandler(self.file_log_handler)

        self.logger.info("Run directory created: %s", self.run_dir)
        self.logger.info("Detailed log file for this run: %s", self.log_file_path)

        self.artifacts = {
            "history_path": os.path.join(self.run_dir, "training_history.json"),
            "hyperparameters_path": os.path.join(self.run_dir, "hyperparameters_and_summary.json"),
            "model_keras_path": os.path.join(self.run_dir, "model_final.keras"),
            "model_h5_path": os.path.join(self.run_dir, "model_final.h5"),
            "saved_model_dir_path": os.path.join(self.run_dir, "model_final_tf_savedmodel"),
            "epoch_models_dir": os.path.join(self.run_dir, "epoch_models"),
        }

    def _initialise_history_manager(self) -> None:
        factory = self.dependencies.get_factory("history_manager")
        self.history_manager = self._invoke_factory(
            factory,
            {"history_path": self.artifacts["history_path"]},
            {"orchestrator": self, "artifacts": self.artifacts},
        )

    def close(self) -> None:
        """Detach file handlers when the orchestrator is no longer needed."""
        root_logger = logging.getLogger()
        if getattr(self, "file_log_handler", None) in root_logger.handlers:
            handler = self.file_log_handler
            root_logger.removeHandler(handler)
            handler.close()

    # ------------------------------------------------------------------
    # Callback helpers for trainers
    # ------------------------------------------------------------------
    def record_epoch_duration(self, duration_seconds: float) -> None:
        """Callback for :class:`Trainer` implementations to report precise timings."""
        self.epoch_durations.append(duration_seconds)

    def save_model_per_epoch(self, epoch_num: int) -> None:
        """Persist a checkpoint for the supplied epoch."""
        if self.model is None:
            self.logger.error("Model unavailable when attempting to save epoch %s", epoch_num)
            return

        os.makedirs(self.artifacts["epoch_models_dir"], exist_ok=True)
        path = os.path.join(self.artifacts["epoch_models_dir"], f"model_epoch_{epoch_num:03d}.keras")
        try:
            self.model.save(path)
            self.logger.info("Checkpoint saved for epoch %s to %s", epoch_num, path)
        except Exception:  # pragma: no cover - defensive logging
            self.logger.exception("Failed to save checkpoint for epoch %s", epoch_num)

    def save_model_all_formats(self) -> None:
        """Persist the trained model in Keras, H5, and SavedModel formats."""
        if self.model is None:
            self.logger.error("Model unavailable; skipping final export.")
            return

        try:
            self.model.save(self.artifacts["model_keras_path"])
            self.model.save(self.artifacts["model_h5_path"])
            self.model.export(self.artifacts["saved_model_dir_path"])
        except Exception:  # pragma: no cover - defensive logging
            self.logger.exception("Model export failed")

    # ------------------------------------------------------------------
    # Hyperparameter utilities
    # ------------------------------------------------------------------
    def save_hyperparameters(self) -> None:
        payload = {
            "run_info": {
                "run_directory": self.run_dir,
                "log_file": self.log_file_path,
            },
            "data_parameters": dict(self.config.data),
            "training_parameters": dict(self.config.training),
            "callback_parameters": dict(self.config.callbacks),
            "model_parameters": dict(self.config.model),
            "metadata": dict(self.config.metadata),
        }

        if self.training_summary:
            payload["training_summary"] = self.training_summary

        try:
            with open(self.artifacts["hyperparameters_path"], "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=4, default=str)
        except Exception:  # pragma: no cover - defensive logging
            self.logger.exception("Failed to persist hyperparameters to %s", self.artifacts["hyperparameters_path"])

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Execute the end-to-end pipeline and return a context dictionary."""
        context: Dict[str, Any] = {"success": False}
        pipeline_start = time.time()

        try:
            if self.hooks.before_data:
                self.hooks.before_data(self)

            data_loader_factory = self.dependencies.get_factory("data_loader")
            data_loader = self._invoke_factory(
                data_loader_factory,
                {"file_path": self.config.file_path},
                {**self.config.data, "orchestrator": self, "artifacts": self.artifacts},
            )
            raw_data = data_loader.get_data()
            data_splits = self._normalise_data_splits(raw_data)

            if self.hooks.after_data:
                self.hooks.after_data(self, data_splits)

            X_train = data_splits.get("X_train")
            if X_train is None or getattr(X_train, "shape", (0,))[0] == 0:
                raise RuntimeError("Training data is empty; aborting pipeline")

            self.save_hyperparameters()

            model_params = {
                "time_steps": data_splits["X_train"].shape[1],
                "num_features": data_splits["X_train"].shape[2],
            }
            model_params.update(self.config.model)

            if self.hooks.before_model_build:
                self.hooks.before_model_build(self, model_params)

            model_builder_factory = self.dependencies.get_factory("model_builder")
            builder = self._invoke_factory(
                model_builder_factory,
                model_params,
                {"orchestrator": self, "artifacts": self.artifacts},
            )
            if hasattr(builder, "build_model"):
                model = builder.build_model()
            elif callable(builder):
                model = builder()
            else:
                model = builder
            self.model = model

            if self.hooks.after_model_build:
                self.hooks.after_model_build(self, model)

            trainer_inputs = {
                "model": model,
                "X_train": data_splits["X_train"],
                "y_train": data_splits["y_train"],
                "X_val": data_splits.get("X_val"),
                "y_val": data_splits.get("y_val"),
                "history_path": self.artifacts["history_path"],
                "main_model_instance": self,
            }
            trainer_inputs.update(dict(self.config.training))
            trainer_inputs.update(dict(self.config.callbacks))

            if self.hooks.before_training:
                self.hooks.before_training(self, trainer_inputs)

            trainer_factory = self.dependencies.get_factory("trainer")
            self.trainer = self._invoke_factory(trainer_factory, trainer_inputs, {"artifacts": self.artifacts})
            history = self.trainer.train()

            if self.hooks.after_training:
                self.hooks.after_training(self, history)

            self._process_training_summary(history)
            self.save_hyperparameters()

            if history and hasattr(history, "history") and self.history_manager is not None:
                self.history_manager.save_history(history)

            if self.hooks.before_evaluation:
                self.hooks.before_evaluation(self)

            evaluator_factory = self.dependencies.get_factory("evaluator")
            self.evaluator = self._invoke_factory(
                evaluator_factory,
                {"model": model},
                {"data": data_splits, "artifacts": self.artifacts, "orchestrator": self},
            )
            evaluation_payload = self._run_evaluation(history)

            if self.hooks.after_evaluation:
                self.hooks.after_evaluation(self, evaluation_payload)

            self.save_model_all_formats()

            context.update(
                {
                    "success": True,
                    "history": history,
                    "evaluation": evaluation_payload,
                    "training_summary": self.training_summary,
                }
            )
            return context
        finally:
            total_time = time.time() - pipeline_start
            context["total_time_seconds"] = total_time
            context.setdefault("success", False)

            if context["success"]:
                self.logger.info("Pipeline completed successfully in %.3f seconds", total_time)
            else:
                self.logger.error("Pipeline failed after %.3f seconds", total_time)

            if self.hooks.finalize:
                self.hooks.finalize(self, context)

            self.close()

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _normalise_data_splits(self, raw_data: Any) -> Dict[str, Any]:
        if isinstance(raw_data, Mapping):
            return dict(raw_data)

        if isinstance(raw_data, Tuple) and len(raw_data) >= 6:
            splits = {
                "X_train": raw_data[0],
                "X_val": raw_data[1],
                "X_test": raw_data[2],
                "y_train": raw_data[3],
                "y_val": raw_data[4],
                "y_test": raw_data[5],
            }
            if len(raw_data) > 6:
                splits["scaler_y"] = raw_data[6]
            return splits

        raise TypeError(
            "DataLoader.get_data() must return either a mapping or the canonical tuple "
            "(X_train, X_val, X_test, y_train, y_val, y_test, [scaler_y])."
        )

    def _process_training_summary(self, history: Any) -> None:
        if not history or not hasattr(history, "history"):
            self.training_summary = {}
            return

        history_dict = getattr(history, "history", {})
        val_losses = list(history_dict.get("val_loss", []))
        best_val_loss = float("inf")
        best_epoch_idx = -1
        if val_losses:
            best_epoch_idx = int(min(range(len(val_losses)), key=val_losses.__getitem__))
            best_val_loss = float(val_losses[best_epoch_idx])

        total_training_time = sum(self.epoch_durations)
        actual_epochs = len(self.epoch_durations)
        time_to_best = None
        if 0 <= best_epoch_idx < actual_epochs:
            time_to_best = sum(self.epoch_durations[: best_epoch_idx + 1])

        self.training_summary = {
            "total_training_time_seconds": round(total_training_time, 3),
            "actual_epochs_run": actual_epochs,
            "best_val_loss_epoch_num": best_epoch_idx + 1 if best_epoch_idx >= 0 else None,
            "best_val_loss_value": None if best_val_loss == float("inf") else round(best_val_loss, 6),
            "time_to_best_val_loss_epoch_seconds": None if time_to_best is None else round(time_to_best, 3),
            "avg_time_per_epoch_seconds": None
            if actual_epochs == 0
            else round(total_training_time / actual_epochs, 3),
            "epoch_durations_sec_list": [round(v, 3) for v in self.epoch_durations],
        }

    def _run_evaluation(self, history: Any) -> Dict[str, Any]:
        if self.evaluator is None:
            return {}

        evaluation_payload: Dict[str, Any] = {}
        evaluator = self.evaluator
        for method_name in [
            "predict",
            "calculate_metrics",
            "save_metrics_to_file",
            "plot_loss",
            "plot_metric_evolution",
            "plot_r2_bar",
            "plot_predictions",
            "plot_error_distribution",
        ]:
            if method_name == "plot_loss" and not (history and hasattr(history, "history")):
                continue

            method = getattr(evaluator, method_name, None)
            if not callable(method):
                continue

            try:
                if method_name == "plot_loss":
                    method(history)
                elif method_name == "plot_metric_evolution":
                    for metric in ["mae", "mse", "val_r2_custom"]:
                        if history and hasattr(history, "history") and metric in history.history:
                            method(history, metric, f"{metric}_evolution_plot.png")
                elif method_name == "plot_r2_bar":
                    for target in ["r2_test", "r2_val"]:
                        score = getattr(evaluator, target, None)
                        if score is not None:
                            method(score, dataset_name="Test" if target.endswith("test") else "Validation")
                else:
                    result = method()
                    if result is not None:
                        evaluation_payload[method_name] = result
            except Exception:  # pragma: no cover - defensive logging
                self.logger.exception("Evaluator step '%s' failed", method_name)

        return evaluation_payload

    def _invoke_factory(
        self,
        factory: Callable[..., Any],
        required_kwargs: Mapping[str, Any],
        optional_kwargs: Mapping[str, Any],
    ) -> Any:
        """Call ``factory`` with keyword arguments filtered by its signature."""

        signature = inspect.signature(factory)
        accepts_var_kw = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
        accepted_names = set(signature.parameters)

        kwargs = dict(required_kwargs)
        if accepts_var_kw:
            kwargs.update(optional_kwargs)
        else:
            for key, value in optional_kwargs.items():
                if key in accepted_names:
                    kwargs[key] = value
        return factory(**kwargs)


__all__ = [
    "DependencyFactoryRegistry",
    "OrchestratorConfig",
    "OrchestratorDependencies",
    "OrchestratorHooks",
    "TimeSeriesOrchestrator",
]
