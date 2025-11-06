"""Helper utilities shared by classical baselines."""

from __future__ import annotations

import logging
from typing import Callable, Mapping, Sequence

import numpy as np

from src.data.dataset import WindowedData
from src.training.engine import EpochMetrics, TrainingSummary

LOGGER = logging.getLogger(__name__)

Forecaster = Callable[[Sequence[float], int], np.ndarray]


def _to_float_list(values: Sequence[float]) -> list[float]:
    return [float(v) for v in values]


def rolling_forecast(
    base_history: Sequence[float],
    partition_targets: np.ndarray,
    lookback: int,
    horizon_steps: int,
    forecaster: Forecaster,
) -> np.ndarray:
    """Generate horizon-step forecasts using a rolling update strategy."""

    series = np.asarray(partition_targets, dtype=np.float64)
    if series.size == 0:
        return np.empty((0,), dtype=np.float32)

    limit = series.shape[0] - lookback - horizon_steps + 1
    if limit <= 0:
        return np.empty((0,), dtype=np.float32)

    history = _to_float_list(base_history)
    history.extend(float(value) for value in series[:lookback])

    predictions = np.empty((limit,), dtype=np.float32)
    for idx in range(limit):
        try:
            forecast = forecaster(history, horizon_steps)
        except Exception:  # pragma: no cover - defensive guard around external libs
            LOGGER.exception("Forecaster %s failed; falling back to naive prediction", forecaster)
            forecast = np.array([history[-1]], dtype=np.float32)

        if len(forecast) == 0:
            predictions[idx] = float(history[-1])
        else:
            predictions[idx] = float(forecast[-1])

        next_obs_index = idx + lookback
        if next_obs_index < series.shape[0]:
            history.append(float(series[next_obs_index]))

    return predictions


def _compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    if predictions.size == 0 or targets.size == 0:
        return float("nan"), float("nan")

    preds = np.asarray(predictions, dtype=np.float64)
    actuals = np.asarray(targets, dtype=np.float64)
    if preds.shape[0] != actuals.shape[0]:
        aligned = min(preds.shape[0], actuals.shape[0])
        LOGGER.warning(
            "Prediction/target length mismatch (%d vs %d); truncating to %d samples",
            preds.shape[0],
            actuals.shape[0],
            aligned,
        )
        preds = preds[:aligned]
        actuals = actuals[:aligned]

    mse = float(np.mean((preds - actuals) ** 2))
    mae = float(np.mean(np.abs(preds - actuals)))
    return mse, mae


def evaluate_forecaster(
    window: WindowedData,
    lookback: int,
    horizon_steps: int,
    forecaster: Forecaster,
) -> tuple[TrainingSummary, dict[str, Mapping[str, float]]]:
    """Evaluate a classical forecaster on train/val/test partitions."""

    train_series = window.train_series
    val_series = window.val_series
    test_series = window.test_series

    train_preds = rolling_forecast([], train_series.targets, lookback, horizon_steps, forecaster)
    train_loss, train_mae = _compute_metrics(train_preds, train_series.sequence_targets)

    val_history = train_series.targets.tolist()
    val_preds = rolling_forecast(val_history, val_series.targets, lookback, horizon_steps, forecaster)
    val_loss, val_mae = _compute_metrics(val_preds, val_series.sequence_targets)

    if val_series.targets.size:
        test_history = np.concatenate([train_series.targets, val_series.targets]).tolist()
    else:
        test_history = train_series.targets.tolist()

    test_preds = rolling_forecast(test_history, test_series.targets, lookback, horizon_steps, forecaster)
    test_loss, test_mae = _compute_metrics(test_preds, test_series.sequence_targets)

    summary = TrainingSummary(
        epochs=[EpochMetrics(train_loss=train_loss, val_loss=val_loss, val_mae=val_mae)],
        best_val_loss=val_loss,
        device="cpu",
    )

    metrics: dict[str, Mapping[str, float]] = {
        "train": {
            "mse": train_loss,
            "mae": train_mae,
            "samples": float(train_series.sequence_targets.shape[0]),
        },
        "val": {
            "mse": val_loss,
            "mae": val_mae,
            "samples": float(val_series.sequence_targets.shape[0]),
        },
        "test": {
            "mse": test_loss,
            "mae": test_mae,
            "samples": float(test_series.sequence_targets.shape[0]),
        },
    }

    return summary, metrics
