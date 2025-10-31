"""Helpers that translate legacy ``MainClass`` wiring into orchestrator factories."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, MutableMapping

from src.core.orchestrator import OrchestratorHooks


@dataclass(frozen=True)
class LegacyDependencyBundle:
    """Container describing factories and hooks for a legacy experiment version."""

    factories: Dict[str, Callable[..., Any]]
    build_hooks: Callable[[], OrchestratorHooks]


def _filter_kwargs(
    factory: Callable[..., Any],
    required: Mapping[str, Any],
    optional: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """Filter ``optional`` kwargs based on ``factory``'s signature."""

    signature = inspect.signature(factory)
    accepts_var_kw = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    kwargs: MutableMapping[str, Any] = dict(required)

    if accepts_var_kw:
        kwargs.update(optional)
    else:
        for key, value in optional.items():
            if key in signature.parameters:
                kwargs[key] = value
    return kwargs


def create_dependency_bundle(version_package: str) -> LegacyDependencyBundle:
    """Build factories that adapt a ``v_*`` package to :class:`TimeSeriesOrchestrator`."""

    data_module = importlib.import_module(f"{version_package}.DataLoader")
    model_module = importlib.import_module(f"{version_package}.ModelBuilder")
    trainer_module = importlib.import_module(f"{version_package}.Trainer")
    evaluator_module = importlib.import_module(f"{version_package}.Evaluator")
    history_module = importlib.import_module(f"{version_package}.HistoryManager")

    DataLoader = getattr(data_module, "DataLoader")
    ModelBuilder = getattr(model_module, "ModelBuilder")
    Trainer = getattr(trainer_module, "Trainer")
    Evaluator = getattr(evaluator_module, "Evaluator")
    HistoryManager = getattr(history_module, "HistoryManager")

    def create_data_loader(
        *,
        file_path: str,
        orchestrator: Any | None = None,
        artifacts: Mapping[str, Any] | None = None,
        **data_kwargs: Any,
    ) -> Any:
        kwargs = _filter_kwargs(DataLoader, {"file_path": file_path}, data_kwargs)
        loader = DataLoader(**kwargs)
        if orchestrator is not None:
            setattr(orchestrator, "data_loader", loader)
        return loader

    def create_model_builder(
        *,
        time_steps: int,
        num_features: int,
        orchestrator: Any | None = None,
        artifacts: Mapping[str, Any] | None = None,
        **model_kwargs: Any,
    ) -> Any:
        kwargs = _filter_kwargs(
            ModelBuilder,
            {"time_steps": time_steps, "num_features": num_features},
            model_kwargs,
        )
        return ModelBuilder(**kwargs)

    def create_trainer(
        *,
        model: Any,
        X_train: Any,
        y_train: Any,
        X_val: Any | None = None,
        y_val: Any | None = None,
        history_path: str,
        orchestrator: Any | None = None,
        artifacts: Mapping[str, Any] | None = None,
        **trainer_kwargs: Any,
    ) -> Any:
        base_kwargs = {
            "model": model,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "history_path": history_path,
            "main_model_instance": orchestrator,
        }
        kwargs = _filter_kwargs(Trainer, base_kwargs, trainer_kwargs)
        return Trainer(**kwargs)

    def create_evaluator(
        *,
        model: Any,
        data: Mapping[str, Any] | None = None,
        orchestrator: Any | None = None,
        artifacts: Mapping[str, Any] | None = None,
        history_manager: Any | None = None,
        **evaluator_kwargs: Any,
    ) -> Any:
        data = dict(data or {})
        base_kwargs = {
            "model": model,
            "X_test": data.get("X_test"),
            "y_test": data.get("y_test"),
            "scaler_y": data.get("scaler_y"),
            "run_dir": getattr(orchestrator, "run_dir", ""),
            "X_val": data.get("X_val"),
            "y_val": data.get("y_val"),
            "history_manager": history_manager or getattr(orchestrator, "history_manager", None),
        }
        kwargs = _filter_kwargs(Evaluator, base_kwargs, evaluator_kwargs)
        return Evaluator(**kwargs)

    def create_history_manager(
        *,
        history_path: str,
        orchestrator: Any | None = None,
        artifacts: Mapping[str, Any] | None = None,
        **history_kwargs: Any,
    ) -> Any:
        kwargs = _filter_kwargs(HistoryManager, {"history_path": history_path}, history_kwargs)
        return HistoryManager(**kwargs)

    def build_hooks() -> OrchestratorHooks:
        def after_training(orchestrator: Any, history: Any) -> None:
            trainer = getattr(orchestrator, "trainer", None)
            callback = getattr(trainer, "epoch_timer_callback", None) if trainer else None
            epoch_durations = getattr(callback, "epoch_durations", None) if callback else None
            if epoch_durations:
                orchestrator.epoch_durations = list(epoch_durations)

        return OrchestratorHooks(after_training=after_training)

    factories: Dict[str, Callable[..., Any]] = {
        "data_loader": create_data_loader,
        "model_builder": create_model_builder,
        "trainer": create_trainer,
        "evaluator": create_evaluator,
        "history_manager": create_history_manager,
    }

    return LegacyDependencyBundle(factories=factories, build_hooks=build_hooks)


__all__ = ["LegacyDependencyBundle", "create_dependency_bundle"]
