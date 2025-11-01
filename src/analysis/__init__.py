"""Analysis utilities exposed at the package level."""

from .benchmark import BenchmarkMetrics, BenchmarkReport, HardwareSpec, benchmark_model, save_report
from .interpretability import (
    AttentionHeatmapResult,
    AttributionResult,
    ExpertTraceResult,
    compute_gradient_attributions,
    generate_attention_heatmaps,
    generate_expert_utilization_trace,
)
from .stats import StatisticalAnalyzer

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
    "save_report",
]
