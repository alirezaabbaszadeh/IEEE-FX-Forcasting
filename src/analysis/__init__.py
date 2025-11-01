"""Analysis utilities exposed at the package level."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import convenience for type checkers
    from .benchmark import BenchmarkMetrics, BenchmarkReport, HardwareSpec, benchmark_model, save_report
    from .hparam import plot_response_curves
    from .interpretability import (
        AttentionHeatmapResult,
        AttributionResult,
        ExpertTraceResult,
        compute_gradient_attributions,
        generate_attention_heatmaps,
        generate_expert_utilization_trace,
    )
    from .stats import StatisticalAnalyzer
else:  # pragma: no cover - optional dependencies may be absent at runtime
    try:
        from .benchmark import BenchmarkMetrics, BenchmarkReport, HardwareSpec, benchmark_model, save_report
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        BenchmarkMetrics = BenchmarkReport = HardwareSpec = None  # type: ignore[assignment]

        def benchmark_model(*args, **kwargs):  # type: ignore[no-redef]
            raise ImportError("Torch is required to use benchmark_model")

        def save_report(*args, **kwargs):  # type: ignore[no-redef]
            raise ImportError("Torch is required to save benchmark reports")

    try:
        from .interpretability import (
            AttentionHeatmapResult,
            AttributionResult,
            ExpertTraceResult,
            compute_gradient_attributions,
            generate_attention_heatmaps,
            generate_expert_utilization_trace,
        )
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        AttentionHeatmapResult = AttributionResult = ExpertTraceResult = None  # type: ignore[assignment]

        def compute_gradient_attributions(*args, **kwargs):  # type: ignore[no-redef]
            raise ImportError("Torch is required for interpretability utilities")

        def generate_attention_heatmaps(*args, **kwargs):  # type: ignore[no-redef]
            raise ImportError("Torch is required for interpretability utilities")

        def generate_expert_utilization_trace(*args, **kwargs):  # type: ignore[no-redef]
            raise ImportError("Torch is required for interpretability utilities")

    try:
        from .stats import StatisticalAnalyzer
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        StatisticalAnalyzer = None  # type: ignore[assignment]

    try:
        from .hparam import plot_response_curves
    except ModuleNotFoundError:  # pragma: no cover - should always be available
        plot_response_curves = None  # type: ignore[assignment]

__all__ = [
    "AttentionHeatmapResult",
    "AttributionResult",
    "BenchmarkMetrics",
    "BenchmarkReport",
    "HardwareSpec",
    "ExpertTraceResult",
    "StatisticalAnalyzer",
    "benchmark_model",
    "compute_gradient_attributions",
    "generate_attention_heatmaps",
    "generate_expert_utilization_trace",
    "plot_response_curves",
    "save_report",
]
