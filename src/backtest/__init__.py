"""Backtesting utilities for simulating strategy performance."""
from __future__ import annotations

from importlib import import_module
from typing import Any

from .costs import CostScenario, apply_costs, apply_costs_to_scenarios

__all__ = [
    "BacktestResult",
    "align_walk_forward_forecasts",
    "append_metrics_to_pbo_table",
    "append_backtest_summary",
    "simulate_strategy_returns",
    "CostScenario",
    "apply_costs",
    "apply_costs_to_scenarios",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin lazy import wrapper
    if name in {
        "BacktestResult",
        "align_walk_forward_forecasts",
        "append_metrics_to_pbo_table",
        "append_backtest_summary",
        "simulate_strategy_returns",
    }:
        module = import_module(".engine", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
