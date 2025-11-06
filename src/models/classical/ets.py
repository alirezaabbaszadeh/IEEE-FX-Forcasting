"""Exponential smoothing baseline aligned with the shared data pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.data.dataset import WindowedData

from .utils import evaluate_forecaster

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as _ExponentialSmoothing
except ModuleNotFoundError:  # pragma: no cover - graceful fallback when statsmodels absent
    _ExponentialSmoothing = None


@dataclass
class ETSConfig:
    """Configuration controlling the exponential smoothing baseline."""

    trend: str | None = "add"
    damped_trend: bool | None = False
    seasonal: str | None = None
    seasonal_periods: int | None = None
    use_boxcox: float | bool | None = None
    initialization_method: str = "estimated"


class ETSBaseline:
    """Wrapper around ``statsmodels``' exponential smoothing implementation."""

    def __init__(self, config: ETSConfig):
        self.config = config

    def __repr__(self) -> str:  # pragma: no cover - convenience for logging only
        return (
            "ETSBaseline(trend={trend}, seasonal={seasonal}, periods={periods})".format(
                trend=self.config.trend,
                seasonal=self.config.seasonal,
                periods=self.config.seasonal_periods,
            )
        )

    def _naive_forecast(self, history: Sequence[float], steps: int) -> np.ndarray:
        last = float(history[-1]) if history else 0.0
        return np.full((steps,), last, dtype=np.float32)

    def forecast(self, history: Sequence[float], steps: int) -> np.ndarray:
        if steps <= 0:
            return np.empty((0,), dtype=np.float32)

        if _ExponentialSmoothing is None:
            LOGGER.warning("statsmodels is unavailable; falling back to naive ETS predictions")
            return self._naive_forecast(history, steps)

        try:
            model_kwargs = {
                "trend": self.config.trend,
                "damped_trend": self.config.damped_trend,
                "seasonal": self.config.seasonal,
                "initialization_method": self.config.initialization_method,
                "use_boxcox": self.config.use_boxcox,
            }
            if self.config.seasonal is not None and self.config.seasonal_periods is not None:
                model_kwargs["seasonal_periods"] = int(self.config.seasonal_periods)

            model = _ExponentialSmoothing(list(history), **model_kwargs)
            results = model.fit()
            forecast = results.forecast(steps)
        except Exception as exc:  # pragma: no cover - defensive guard against library errors
            LOGGER.warning("ETS forecast failed (%s); using naive fallback", exc)
            return self._naive_forecast(history, steps)

        return np.asarray(forecast, dtype=np.float32)


def run_ets_baseline(
    window: WindowedData,
    *,
    lookback: int,
    horizon_steps: int,
    config: ETSConfig | None = None,
):
    """Evaluate an ETS baseline on the provided walk-forward window."""

    cfg = config or ETSConfig()
    baseline = ETSBaseline(cfg)
    summary, metrics = evaluate_forecaster(window, lookback, horizon_steps, baseline.forecast)
    return summary, metrics


__all__ = ["ETSConfig", "ETSBaseline", "run_ets_baseline"]
