"""Backtest execution helpers with purged walk-forward alignment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import math
import pandas as pd

from src.backtest.costs import CostScenario, apply_costs_to_scenarios
from src.splits.walk_forward import WalkForwardSplit


@dataclass(frozen=True)
class BacktestResult:
    """Container exposing strategy returns and diagnostic summaries."""

    positions: pd.Series
    gross_returns: pd.Series
    net_returns: pd.DataFrame
    turnover: pd.Series
    annualisation_factor: int = 252

    def summary_frame(
        self,
        *,
        run_id: str | None = None,
        pair: str | None = None,
        horizon: object | None = None,
    ) -> pd.DataFrame:
        """Return a tidy dataframe describing backtest statistics."""

        summaries = []
        for scenario, returns in self.iter_returns().items():
            summaries.extend(
                _summarise_returns(
                    scenario,
                    returns,
                    turnover=self.turnover,
                    annualisation_factor=self.annualisation_factor,
                )
            )
        frame = pd.DataFrame(summaries)
        frame["run_id"] = run_id
        frame["pair"] = pair
        frame["horizon"] = horizon
        columns = ["run_id", "pair", "horizon", "scenario", "metric", "value"]
        return frame[columns]

    def iter_returns(self) -> Mapping[str, pd.Series]:
        """Iterate over the gross and net return series."""

        data: dict[str, pd.Series] = {"gross": self.gross_returns}
        for column in self.net_returns.columns:
            data[column] = self.net_returns[column]
        return data


def align_walk_forward_forecasts(
    forecasts: pd.Series | pd.DataFrame,
    realised: pd.Series,
    splits: Sequence[WalkForwardSplit],
    *,
    timestamp_col: str = "timestamp",
    value_col: str = "forecast",
) -> pd.Series:
    """Align forecasts to realised returns using purged walk-forward splits."""

    realised_index = pd.Index(realised.index)
    eligible_timestamps: list[pd.Timestamp] = []
    for split in splits:
        window = split.window
        timestamps = realised_index[window.test]
        eligible_timestamps.extend(timestamps)
    ordered = list(dict.fromkeys(eligible_timestamps))
    eligible_index = pd.Index(ordered)

    if isinstance(forecasts, pd.Series):
        forecast_series = forecasts
    else:
        if timestamp_col not in forecasts or value_col not in forecasts:
            raise KeyError("Forecast frame must contain timestamp and value columns")
        forecast_series = forecasts.set_index(timestamp_col)[value_col]

    aligned = forecast_series.reindex(eligible_index)
    aligned = aligned.dropna()
    aligned.name = value_col
    return aligned


def _ensure_series(data: pd.Series | pd.DataFrame, value_col: str = "forecast") -> pd.Series:
    if isinstance(data, pd.Series):
        return data.astype(float)
    if value_col not in data:
        raise KeyError(f"Column '{value_col}' not found in forecasts frame")
    return data[value_col].astype(float)


def _compute_gross_returns(positions: pd.Series, realised: pd.Series) -> pd.Series:
    aligned_realised = realised.reindex(positions.index)
    if aligned_realised.isna().any():
        missing = aligned_realised[aligned_realised.isna()].index
        raise KeyError(f"Realised returns missing for timestamps: {missing.tolist()}")
    gross = positions * aligned_realised
    gross.name = "gross"
    return gross


def _compute_turnover(positions: pd.Series) -> pd.Series:
    turnover = positions.diff().abs()
    if not turnover.empty:
        turnover.iloc[0] = abs(float(positions.iloc[0]))
    return turnover.fillna(0.0)


def simulate_strategy_returns(
    forecasts: pd.Series | pd.DataFrame,
    realised: pd.Series,
    *,
    splits: Sequence[WalkForwardSplit] | None = None,
    cost_scenarios: Iterable[CostScenario] | Mapping[str, CostScenario] | None = None,
    annualisation_factor: int = 252,
    value_col: str = "forecast",
) -> BacktestResult:
    """Simulate strategy returns from forecasts and realised prices."""

    if splits is not None:
        forecast_series = align_walk_forward_forecasts(
            forecasts, realised, splits, value_col=value_col
        )
    else:
        forecast_series = _ensure_series(forecasts, value_col=value_col)
        forecast_series = forecast_series.reindex(realised.index).dropna()

    positions = forecast_series.astype(float)
    positions.name = "position"

    gross_returns = _compute_gross_returns(positions, realised)
    turnover = _compute_turnover(positions)

    scenarios = _normalise_scenarios(cost_scenarios)
    net_returns = apply_costs_to_scenarios(gross_returns, positions, scenarios.values())

    return BacktestResult(
        positions=positions,
        gross_returns=gross_returns,
        net_returns=net_returns,
        turnover=turnover,
        annualisation_factor=annualisation_factor,
    )


def _normalise_scenarios(
    scenarios: Iterable[CostScenario] | Mapping[str, CostScenario] | None,
) -> Mapping[str, CostScenario]:
    if scenarios is None:
        return {}
    if isinstance(scenarios, Mapping):
        return dict(scenarios)
    scenario_list = list(scenarios)
    scenario_map = {scenario.name: scenario for scenario in scenario_list}
    if len(scenario_map) != len(scenario_list):
        raise ValueError("Scenario names must be unique")
    return scenario_map


def _summarise_returns(
    scenario: str,
    returns: pd.Series,
    *,
    turnover: pd.Series,
    annualisation_factor: int,
) -> list[dict[str, object]]:
    if returns.empty:
        return []
    stats: list[dict[str, object]] = []
    cumulative = float((1.0 + returns).prod() - 1.0)
    mean = float(returns.mean())
    volatility = float(returns.std(ddof=0))
    sharpe = float("nan")
    if volatility > 0.0:
        sharpe = mean / volatility * math.sqrt(annualisation_factor)
    avg_turnover = float(turnover.reindex(returns.index).mean())
    stats.append({"scenario": scenario, "metric": "total_return", "value": cumulative})
    stats.append({"scenario": scenario, "metric": "mean_return", "value": mean})
    stats.append({"scenario": scenario, "metric": "volatility", "value": volatility})
    stats.append({"scenario": scenario, "metric": "sharpe", "value": sharpe})
    stats.append({"scenario": scenario, "metric": "avg_turnover", "value": avg_turnover})
    return stats


def append_metrics_to_pbo_table(
    metrics: pd.DataFrame,
    *,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Append metrics to the consolidated paper PBO table."""

    path = output_path or Path("paper_outputs") / "pbo_table.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    required = {"run_id", "pair", "horizon", "scenario", "metric", "value"}
    missing = required.difference(metrics.columns)
    if missing:
        raise KeyError(f"Metrics frame missing required columns: {sorted(missing)}")
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, metrics], ignore_index=True)
    else:
        combined = metrics.copy()
    combined = combined.drop_duplicates(
        subset=["run_id", "pair", "horizon", "scenario", "metric"], keep="last"
    )
    combined.to_csv(path, index=False)
    return combined


def append_backtest_summary(
    result: BacktestResult,
    *,
    run_id: str,
    pair: str,
    horizon: object,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Append a backtest summary to the consolidated PBO table."""

    summary = result.summary_frame(run_id=run_id, pair=pair, horizon=horizon)
    return append_metrics_to_pbo_table(summary, output_path=output_path)
