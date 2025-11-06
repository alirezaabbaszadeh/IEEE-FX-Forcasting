"""Trading cost utilities for backtest simulations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    import pandas as pd


@dataclass(frozen=True)
class CostScenario:
    """Description of a trading cost scenario.

    Parameters
    ----------
    name:
        Identifier used in reporting outputs.
    commission_bps:
        Broker commission charged in basis points per unit of turnover.
    slippage_bps:
        Ad-hoc execution slippage in basis points per unit of turnover.
    spread_bps:
        Effective bid/ask spread paid per unit of turnover in basis points.
    fixed_cost:
        Optional constant cost applied whenever the strategy trades. The
        value should be expressed in the same return units as the input
        series (e.g. a proportion of notional PnL).
    """

    name: str
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    spread_bps: float = 0.0
    fixed_cost: float = 0.0

    @property
    def total_bps(self) -> float:
        """Return the aggregate proportional cost in basis points."""

        return float(self.commission_bps + self.slippage_bps + self.spread_bps)


def _normalise_series(values: "pd.Series" | Mapping[object, float]) -> "pd.Series":
    import pandas as pd

    if isinstance(values, pd.Series):
        return values.astype(float)
    return pd.Series(values, dtype=float)


def _compute_turnover(positions: "pd.Series") -> "pd.Series":
    import pandas as pd

    """Return absolute turnover implied by the provided position series."""

    diffs = positions.diff().abs()
    if not diffs.empty:
        diffs.iloc[0] = abs(float(positions.iloc[0]))
    return diffs.fillna(0.0)


def apply_costs(
    gross_returns: "pd.Series",
    positions: "pd.Series",
    scenario: CostScenario,
) -> "pd.Series":
    """Apply trading costs to a stream of gross returns.

    Parameters
    ----------
    gross_returns:
        Series of gross strategy returns indexed by timestamp.
    positions:
        Series describing the position taken at each timestamp. The index
        must align with ``gross_returns``. Values are interpreted as the
        fraction of capital deployed (e.g. +1 for fully long, -1 for fully
        short).
    scenario:
        Trading cost configuration to apply.

    Returns
    -------
    pandas.Series
        Net returns after deducting proportional and fixed trading costs.
    """

    if gross_returns.empty:
        return gross_returns.copy()

    import pandas as pd

    gross = _normalise_series(gross_returns)
    aligned_positions = _normalise_series(positions).reindex(gross.index).fillna(0.0)

    turnover = _compute_turnover(aligned_positions)
    proportional_cost = turnover * (scenario.total_bps / 10_000.0)

    if scenario.fixed_cost:
        trade_events = (turnover > 0.0).astype(float)
        proportional_cost += trade_events * scenario.fixed_cost

    net_returns = gross - proportional_cost
    net_returns.name = scenario.name
    return net_returns


def apply_costs_to_scenarios(
    gross_returns: "pd.Series",
    positions: "pd.Series",
    scenarios: Iterable[CostScenario],
) -> "pd.DataFrame":
    """Evaluate multiple trading cost scenarios.

    Parameters
    ----------
    gross_returns:
        Series of gross strategy returns.
    positions:
        Position series aligned with ``gross_returns``.
    scenarios:
        Iterable of :class:`CostScenario` objects.

    Returns
    -------
    pandas.DataFrame
        Columns correspond to scenario names and rows to timestamps.
    """

    import pandas as pd

    scenario_list = list(scenarios)
    if not scenario_list:
        return pd.DataFrame(index=gross_returns.index.copy())

    frames = [apply_costs(gross_returns, positions, scenario) for scenario in scenario_list]
    data = pd.concat(frames, axis=1)
    data.index = gross_returns.index
    return data
