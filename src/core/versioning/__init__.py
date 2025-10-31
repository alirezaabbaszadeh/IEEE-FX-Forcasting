"""Version-specific dependency helpers for the orchestrator."""

from .legacy import LegacyDependencyBundle, create_dependency_bundle

__all__ = ["LegacyDependencyBundle", "create_dependency_bundle"]
