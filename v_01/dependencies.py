"""Dependency registration for v_01."""

from __future__ import annotations

from typing import Dict

from src.core.orchestrator import DependencyFactoryRegistry
from src.core.versioning import create_dependency_bundle

BUNDLE = create_dependency_bundle("v_01")

FACTORIES = BUNDLE.factories
build_hooks = BUNDLE.build_hooks

create_data_loader = FACTORIES["data_loader"]
create_model_builder = FACTORIES["model_builder"]
create_trainer = FACTORIES["trainer"]
create_evaluator = FACTORIES["evaluator"]
create_history_manager = FACTORIES["history_manager"]

REGISTRY_KEYS: Dict[str, str] = {name: name for name in FACTORIES}


def register_factories(registry: DependencyFactoryRegistry) -> DependencyFactoryRegistry:
    """Register the version-specific factories with a registry."""

    for name, factory in FACTORIES.items():
        registry.register(name, factory)
    return registry


__all__ = [
    "FACTORIES",
    "REGISTRY_KEYS",
    "build_hooks",
    "create_data_loader",
    "create_model_builder",
    "create_trainer",
    "create_evaluator",
    "create_history_manager",
    "register_factories",
]
