"""Purged conformal calibration utilities for walk-forward evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedConformalConfig:
    """Configuration options for purged conformal calibration."""

    alpha: float = 0.1
    embargo: int = 0
    calibration_splits: Sequence[str] = ("val",)
    include_past_windows: bool = True
    weight_decay: float | None = None
    min_calibration: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < float(self.alpha) < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if int(self.embargo) < 0:
            raise ValueError("embargo cannot be negative")
        if not self.calibration_splits:
            raise ValueError("at least one calibration split must be provided")
        if self.weight_decay is not None:
            decay = float(self.weight_decay)
            if not 0.0 < decay <= 1.0:
                raise ValueError("weight_decay must lie in (0, 1]")
        if int(self.min_calibration) <= 0:
            raise ValueError("min_calibration must be a positive integer")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "PurgedConformalConfig":
        """Instantiate the configuration from a mapping or OmegaConf object."""

        data = dict(payload)
        if "calibration_splits" in data:
            data["calibration_splits"] = tuple(str(item) for item in data["calibration_splits"])
        return cls(**data)


class PurgedConformalCalibrator:
    """Generate predictive intervals with embargo-aware calibration sets."""

    REQUIRED_COLUMNS = {
        "pair",
        "horizon",
        "window_id",
        "split",
        "timestamp",
        "y_true",
        "y_pred",
    }

    def __init__(self, config: PurgedConformalConfig):
        self.cfg = config

    def calibrate(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Return test rows augmented with conformal prediction intervals."""

        frame = self._prepare_frame(predictions)
        results: list[pd.DataFrame] = []
        group_columns = ["pair", "horizon"]

        for (pair, horizon), group in frame.groupby(group_columns, sort=False):
            group_sorted = group.sort_values(["timestamp", "window_id", "split"]).copy()
            group_sorted["_order"] = np.arange(len(group_sorted), dtype=int)
            calibrated = self._calibrate_group(pair, horizon, group_sorted)
            if calibrated is not None and not calibrated.empty:
                results.append(calibrated)

        if not results:
            return pd.DataFrame(columns=list(predictions.columns) + self._interval_columns())
        return pd.concat(results, ignore_index=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_frame(self, predictions: pd.DataFrame) -> pd.DataFrame:
        missing = self.REQUIRED_COLUMNS.difference(predictions.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

        frame = predictions.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        if frame["timestamp"].isnull().any():
            raise ValueError("Timestamp column contains non-parsable values")
        frame["window_id"] = frame["window_id"].astype(int)
        frame["split"] = frame["split"].astype(str)
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        return frame

    def _calibrate_group(
        self, pair: str, horizon: object, group: pd.DataFrame
    ) -> pd.DataFrame | None:
        outputs: list[pd.DataFrame] = []
        for window_id in sorted(group["window_id"].unique()):
            test_mask = (group["window_id"] == window_id) & (group["split"].str.lower() == "test")
            if not test_mask.any():
                continue
            test_rows = group.loc[test_mask].copy()
            test_start_order = int(test_rows["_order"].min())
            calibration = self._calibration_rows(group, window_id, test_start_order)
            if calibration.empty:
                raise ValueError(
                    f"Insufficient calibration data for pair={pair!r}, horizon={horizon!r},"
                    f" window={window_id}."
                )
            if len(calibration) < int(self.cfg.min_calibration):
                raise ValueError(
                    "Calibration set smaller than min_calibration; increase available data or"
                    " relax constraints"
                )

            radius, weights = self._calibration_radius(calibration)
            last_timestamp = calibration["timestamp"].max()
            test_rows = test_rows.copy()
            test_rows["interval_lower"] = test_rows["y_pred"] - radius
            test_rows["interval_upper"] = test_rows["y_pred"] + radius
            test_rows["calibration_radius"] = radius
            test_rows["calibration_size"] = int(calibration.shape[0])
            test_rows["calibration_weight_sum"] = float(weights.sum())
            test_rows["calibration_last_timestamp"] = last_timestamp
            test_rows["alpha"] = float(self.cfg.alpha)
            outputs.append(test_rows.drop(columns=["_order"]))

        if outputs:
            return pd.concat(outputs, ignore_index=True)
        return None

    def _calibration_rows(
        self, group: pd.DataFrame, window_id: int, test_start_order: int
    ) -> pd.DataFrame:
        split_values = {split.lower() for split in self.cfg.calibration_splits}
        current_mask = (group["window_id"] == window_id) & group["split"].str.lower().isin(split_values)
        if self.cfg.include_past_windows:
            previous_mask = (group["window_id"] < window_id) & (
                group["split"].str.lower() == "test"
            )
        else:
            previous_mask = pd.Series(False, index=group.index)
        calibration = group.loc[current_mask | previous_mask].copy()
        if calibration.empty:
            return calibration

        if self.cfg.embargo > 0:
            cutoff = test_start_order - int(self.cfg.embargo)
            calibration = calibration[calibration["_order"] <= cutoff]
        return calibration

    def _calibration_radius(self, calibration: pd.DataFrame) -> tuple[float, np.ndarray]:
        residuals = np.abs(calibration["y_true"].to_numpy(dtype=float) - calibration["y_pred"].to_numpy(dtype=float))
        if residuals.ndim != 1:
            residuals = residuals.reshape(-1)
        weights = self._build_weights(calibration)
        quantile = 1.0 - float(self.cfg.alpha)
        radius = _weighted_quantile(residuals, quantile, sample_weight=weights)
        return float(radius), weights

    def _build_weights(self, calibration: pd.DataFrame) -> np.ndarray:
        n = int(calibration.shape[0])
        if self.cfg.weight_decay is None:
            return np.ones(n, dtype=float)
        order = calibration["_order"].to_numpy(dtype=int)
        max_order = order.max() if order.size else 0
        distances = max_order - order
        decay = float(self.cfg.weight_decay)
        weights = np.power(decay, distances.astype(float))
        return weights.astype(float)

    @staticmethod
    def _interval_columns() -> list[str]:
        return [
            "interval_lower",
            "interval_upper",
            "calibration_radius",
            "calibration_size",
            "calibration_weight_sum",
            "calibration_last_timestamp",
            "alpha",
        ]


def _weighted_quantile(values: Iterable[float], quantile: float, sample_weight: np.ndarray | None = None) -> float:
    """Return the weighted quantile using a right-continuous empirical CDF."""

    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    if array.ndim != 1:
        array = array.reshape(-1)
    if array.size == 0:
        raise ValueError("Cannot compute a quantile over an empty array")

    q = float(np.clip(quantile, 0.0, 1.0))
    if sample_weight is None:
        return float(np.quantile(array, q, method="higher" if q == 1.0 else "linear"))

    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape != array.shape:
        raise ValueError("sample_weight must match the shape of values")
    if np.any(weights < 0):
        raise ValueError("sample_weight cannot contain negative values")
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError("Sum of sample_weight must be positive")

    sorter = np.argsort(array)
    sorted_values = array[sorter]
    sorted_weights = weights[sorter]
    cumulative = np.cumsum(sorted_weights)
    threshold = q * total_weight
    idx = int(np.searchsorted(cumulative, threshold, side="right"))
    if idx >= sorted_values.size:
        idx = sorted_values.size - 1
    return float(sorted_values[idx])
