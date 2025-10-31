"""Legacy compatibility shim that delegates to the shared loader."""
from __future__ import annotations

from typing import Any

from src.data.loader import ChronologicalDataLoader, LegacyLoaderOptions

LEGACY_LOADER_OPTIONS = LegacyLoaderOptions()


class DataLoader(ChronologicalDataLoader):
    """Expose the shared :class:`ChronologicalDataLoader` under the legacy name."""

    def __init__(
        self,
        file_path: str,
        time_steps: int = 3,
        train_ratio: float = 0.94,
        val_ratio: float = 0.03,
        test_ratio: float = 0.03,
        **overrides: Any,
    ) -> None:
        super().__init__(
            file_path,
            time_steps=time_steps,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            options=LEGACY_LOADER_OPTIONS,
            **overrides,
        )


__all__ = ["DataLoader", "LEGACY_LOADER_OPTIONS"]
