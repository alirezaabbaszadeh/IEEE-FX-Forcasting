"""Experiment orchestration and search utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time convenience for type checkers
    from .pcc import run_pcc_toggle
    from .runner import MultiRunExperiment, MultiRunResult, VariantRunCollection
    from .search import HyperparameterSearch
    from .tuning import HyperparameterTuner
else:  # pragma: no cover - exercised implicitly in runtime usage
    try:
        from .pcc import run_pcc_toggle
        from .runner import MultiRunExperiment, MultiRunResult, VariantRunCollection
    except ModuleNotFoundError:  # pragma: no cover - depends on optional deps
        MultiRunExperiment = MultiRunResult = VariantRunCollection = None  # type: ignore[assignment]
        run_pcc_toggle = None  # type: ignore[assignment]
    try:
        from .search import HyperparameterSearch
    except ModuleNotFoundError:  # pragma: no cover - depends on optional deps
        HyperparameterSearch = None  # type: ignore[assignment]
    try:
        from .tuning import HyperparameterTuner
    except ModuleNotFoundError:  # pragma: no cover - depends on optional deps
        HyperparameterTuner = None  # type: ignore[assignment]

__all__ = [
    "MultiRunExperiment",
    "MultiRunResult",
    "VariantRunCollection",
    "HyperparameterSearch",
    "HyperparameterTuner",
    "run_pcc_toggle",
]
