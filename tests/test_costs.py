from __future__ import annotations

import pytest

from src.backtest.costs import CostScenario, apply_costs, apply_costs_to_scenarios

pd = pytest.importorskip("pandas")


def test_apply_costs_accounts_for_turnover_and_fixed_costs() -> None:
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    realised = pd.Series([0.01, 0.02, -0.01], index=index)
    positions = pd.Series([1.0, -1.0, -1.0], index=index)

    scenario = CostScenario(name="base", commission_bps=10.0, slippage_bps=5.0, fixed_cost=0.0005)

    gross = positions * realised
    net = apply_costs(gross, positions, scenario)

    turnover = pd.Series([1.0, 2.0, 0.0], index=index)
    proportional = turnover * ((scenario.commission_bps + scenario.slippage_bps) / 10_000.0)
    fixed = pd.Series([0.0005, 0.0005, 0.0], index=index)
    expected = gross - proportional - fixed

    pd.testing.assert_series_equal(net, expected, check_names=False)


def test_apply_costs_to_scenarios_returns_dataframe_with_named_columns() -> None:
    index = pd.RangeIndex(4)
    gross = pd.Series([0.01, -0.02, 0.015, 0.01], index=index)
    positions = pd.Series([0.0, 0.5, -0.5, 0.0], index=index)

    scenarios = [
        CostScenario(name="tight", commission_bps=5.0),
        CostScenario(name="wide", commission_bps=20.0, slippage_bps=10.0),
    ]

    table = apply_costs_to_scenarios(gross, positions, scenarios)

    assert list(table.columns) == ["tight", "wide"]
    assert table.index.equals(index)
    assert (table <= gross.max()).all().all()
