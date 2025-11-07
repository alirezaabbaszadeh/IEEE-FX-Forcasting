"""Volatility regime labelling utilities for quantile-aware models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

__all__ = [
    "VolatilityRegimeConfig",
    "compute_regime_features",
    "label_volatility_regimes",
]


@dataclass(frozen=True)
class VolatilityRegimeConfig:
    """Configuration governing volatility regime detection."""

    window: int = 96
    min_periods: int = 24
    calm_quantile: float = 0.25
    stress_quantile: float = 0.85
    tail_quantile: float = 0.975
    smoothing_span: int = 32
    stress_zscore: float = 2.5
    clip_zscore: float = 6.0

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be a positive integer")
        if self.min_periods <= 0:
            raise ValueError("min_periods must be a positive integer")
        if self.min_periods > self.window:
            raise ValueError("min_periods cannot exceed window")
        for name, value in {
            "calm_quantile": self.calm_quantile,
            "stress_quantile": self.stress_quantile,
            "tail_quantile": self.tail_quantile,
        }.items():
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")
        if self.calm_quantile >= self.stress_quantile:
            raise ValueError("calm_quantile must be less than stress_quantile")
        if self.clip_zscore <= 0:
            raise ValueError("clip_zscore must be positive")
        if self.stress_zscore <= 0:
            raise ValueError("stress_zscore must be positive")


def _ensure_series(values: Iterable[float] | pd.Series | np.ndarray) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.astype(float)
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=float)
    elif isinstance(values, (list, tuple)):
        array = np.asarray(values, dtype=float)
    else:
        array = np.fromiter(values, dtype=float)
    array = np.atleast_1d(array)
    return pd.Series(array)


def compute_regime_features(
    values: Iterable[float] | pd.Series | np.ndarray,
    *,
    config: VolatilityRegimeConfig | None = None,
) -> pd.DataFrame:
    """Return smoothed volatility and stress features for the provided values."""

    if config is None:
        config = VolatilityRegimeConfig()
    series = _ensure_series(values)
    abs_returns = series.abs()

    squared = abs_returns.pow(2)
    realised_variance = squared.rolling(config.window, min_periods=config.min_periods).mean()
    realised_volatility = realised_variance.pow(0.5)

    smoothed_volatility = realised_volatility.ewm(
        span=config.smoothing_span, min_periods=config.min_periods
    ).mean()
    tail_risk = abs_returns.rolling(config.window, min_periods=config.min_periods).quantile(
        config.tail_quantile
    )
    tail_risk = tail_risk.ewm(span=config.smoothing_span, min_periods=config.min_periods).mean()

    baseline_mean = abs_returns.ewm(span=config.smoothing_span, min_periods=config.min_periods).mean()
    baseline_std = abs_returns.ewm(span=config.smoothing_span, min_periods=config.min_periods).std(bias=False)
    baseline_std = baseline_std.replace(0.0, np.nan)
    stress_score = (abs_returns - baseline_mean) / baseline_std
    stress_score = stress_score.clip(-config.clip_zscore, config.clip_zscore)

    return pd.DataFrame(
        {
            "abs_return": abs_returns,
            "realised_volatility": realised_volatility,
            "smoothed_volatility": smoothed_volatility,
            "tail_risk": tail_risk,
            "stress_score": stress_score,
        },
        index=series.index,
    )


def _fallback_quantile_labels(series: pd.Series, config: VolatilityRegimeConfig) -> pd.Series:
    values = series.abs().dropna()
    if values.empty:
        return pd.Series(["unknown"] * len(series), index=series.index, dtype="object")
    calm, stress = np.quantile(values, [config.calm_quantile, config.stress_quantile])
    labels = pd.Series("volatile", index=series.index, dtype="object")
    labels[series.abs() <= calm] = "calm"
    labels[series.abs() >= stress] = "stress"
    labels[series.isna()] = "unknown"
    return labels


def label_volatility_regimes(
    values: Iterable[float] | pd.Series | np.ndarray,
    *,
    config: VolatilityRegimeConfig | None = None,
) -> pd.Series:
    """Assign calm/volatile/stress regimes using rolling volatility diagnostics."""

    if config is None:
        config = VolatilityRegimeConfig()
    series = _ensure_series(values)
    features = compute_regime_features(series, config=config)
    volatility = features["smoothed_volatility"].ffill()

    valid_volatility = volatility.dropna()
    if valid_volatility.empty:
        return _fallback_quantile_labels(series, config)

    calm_threshold = valid_volatility.quantile(config.calm_quantile)
    stress_threshold = valid_volatility.quantile(config.stress_quantile)

    labels = pd.Series("volatile", index=series.index, dtype="object")
    labels[volatility <= calm_threshold] = "calm"

    stress_mask = (
        (volatility >= stress_threshold)
        | (features["stress_score"].fillna(0.0) >= config.stress_zscore)
        | (series.abs() >= features["tail_risk"].fillna(np.inf))
    )
    labels[stress_mask] = "stress"
    labels[volatility.isna()] = "unknown"
    return labels
