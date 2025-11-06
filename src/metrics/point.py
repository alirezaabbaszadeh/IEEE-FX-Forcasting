"""Point forecasting error metrics."""
from __future__ import annotations

import numpy as np

_EPSILON = 1e-8


def _to_1d_array(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Convert inputs to a 1D float NumPy array."""

    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim > 1:
        return array.reshape(-1)
    return array


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the mean absolute error between `y_true` and `y_pred`."""

    true = _to_1d_array(y_true)
    pred = _to_1d_array(y_pred)
    errors = pred - true
    return float(np.mean(np.abs(errors)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the root-mean-square error between `y_true` and `y_pred`."""

    true = _to_1d_array(y_true)
    pred = _to_1d_array(y_pred)
    errors = pred - true
    return float(np.sqrt(np.mean(np.square(errors))))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the symmetric mean absolute percentage error."""

    true = _to_1d_array(y_true)
    pred = _to_1d_array(y_pred)
    numerator = 2.0 * np.abs(pred - true)
    denominator = np.abs(true) + np.abs(pred)
    return float(np.mean(numerator / np.maximum(denominator, _EPSILON)))


def mase(y_true: np.ndarray, y_pred: np.ndarray, *, seasonality: int = 1) -> float:
    """Return the mean absolute scaled error using in-sample naive forecasts."""

    if seasonality < 1:
        raise ValueError("`seasonality` must be a positive integer")
    true = _to_1d_array(y_true)
    pred = _to_1d_array(y_pred)
    if true.size != pred.size:
        raise ValueError("`y_true` and `y_pred` must be the same length")
    if true.size <= seasonality:
        return float("nan")
    naive_forecasts = true[:-seasonality]
    naive_actuals = true[seasonality:]
    scale = np.mean(np.abs(naive_actuals - naive_forecasts))
    if np.isclose(scale, 0.0):
        return float("nan")
    errors = np.abs(pred - true)
    return float(np.mean(errors) / scale)


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute a core set of point forecast metrics."""

    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mase": mase(y_true, y_pred),
    }


__all__ = ["mae", "rmse", "smape", "mase", "point_metrics"]
