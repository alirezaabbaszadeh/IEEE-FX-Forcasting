"""ARIMA-based baseline leveraging the shared preprocessing pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.data.dataset import WindowedData

from .utils import evaluate_forecaster

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from statsmodels.tsa.arima.model import ARIMA as _StatsmodelsARIMA
except ModuleNotFoundError:  # pragma: no cover - graceful fallback when statsmodels absent
    _StatsmodelsARIMA = None


@dataclass
class ArimaConfig:
    """Configuration parameters controlling the ARIMA baseline."""

    order: tuple[int, int, int] = (1, 1, 0)
    seasonal_order: tuple[int, int, int, int] | None = None
    trend: str | None = None
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True
    method: str | None = None
    maxiter: int | None = 200


class ArimaBaseline:
    """Lightweight wrapper around ``statsmodels``' ARIMA implementation."""

    def __init__(self, config: ArimaConfig):
        self.config = config

    def __repr__(self) -> str:  # pragma: no cover - convenience for logging only
        return f"ArimaBaseline(order={self.config.order}, seasonal_order={self.config.seasonal_order})"

    def _naive_forecast(self, history: Sequence[float], steps: int) -> np.ndarray:
        last = float(history[-1]) if history else 0.0
        return np.full((steps,), last, dtype=np.float32)

    def forecast(self, history: Sequence[float], steps: int) -> np.ndarray:
        if steps <= 0:
            return np.empty((0,), dtype=np.float32)

        if _StatsmodelsARIMA is None:
            LOGGER.warning("statsmodels is unavailable; falling back to naive ARIMA predictions")
            return self._naive_forecast(history, steps)

        model_kwargs = {
            "order": self.config.order,
            "trend": self.config.trend,
            "enforce_stationarity": self.config.enforce_stationarity,
            "enforce_invertibility": self.config.enforce_invertibility,
        }
        if self.config.seasonal_order is not None:
            model_kwargs["seasonal_order"] = self.config.seasonal_order

        try:
            model = _StatsmodelsARIMA(list(history), **model_kwargs)
            fit_kwargs: dict[str, object] = {}
            if self.config.method is not None:
                fit_kwargs["method"] = self.config.method
            if self.config.maxiter is not None:
                fit_kwargs["maxiter"] = int(self.config.maxiter)
            try:
                results = model.fit(disp=0, **fit_kwargs)
            except TypeError:
                results = model.fit(**fit_kwargs)
            forecast = results.forecast(steps=steps)
        except Exception as exc:  # pragma: no cover - defensive guard against library errors
            LOGGER.warning("ARIMA forecast failed (%s); using naive fallback", exc)
            return self._naive_forecast(history, steps)

        return np.asarray(forecast, dtype=np.float32)


def run_arima_baseline(
    window: WindowedData,
    *,
    lookback: int,
    horizon_steps: int,
    config: ArimaConfig | None = None,
):
    """Evaluate an ARIMA baseline on the provided walk-forward window."""

    cfg = config or ArimaConfig()
    baseline = ArimaBaseline(cfg)
    summary, metrics = evaluate_forecaster(window, lookback, horizon_steps, baseline.forecast)
    return summary, metrics


__all__ = ["ArimaBaseline", "ArimaConfig", "run_arima_baseline"]
