
"""Evaluation runner that aggregates metrics across pairs and horizons."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"pair", "horizon", "timestamp", "y_true", "y_pred"}


def _parse_horizon(value: object) -> str:
    """Normalise horizon representations for grouping."""

    try:
        td = pd.to_timedelta(str(value).lower())
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(td):
        return str(value)
    return str(td)


def _ensure_timezone(series: pd.Series, target_tz: str) -> pd.Series:
    timestamps = pd.to_datetime(series, utc=True, errors="coerce")
    if timestamps.isnull().any():
        raise ValueError("Timestamp column contains non-parsable values")
    timestamps = timestamps.dt.tz_convert(target_tz)
    return timestamps


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    signs_true = np.sign(y_true)
    signs_pred = np.sign(y_pred)
    return float(np.mean(signs_true == signs_pred))


def _base_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    mape = float(np.mean(np.abs(errors) / np.maximum(np.abs(y_true), 1e-8)))
    da = _directional_accuracy(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "mape": mape, "directional_accuracy": da}


def _volatility_regime_labels(values: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(np.abs(values), [0.33, 0.66])
    low, high = quantiles
    labels = []
    for val in np.abs(values):
        if val <= low:
            labels.append("low")
        elif val <= high:
            labels.append("medium")
        else:
            labels.append("high")
    return np.array(labels)


def _session_labels(timestamps: pd.Series) -> np.ndarray:
    hours = timestamps.dt.hour
    labels: List[str] = []
    for hour in hours:
        if 0 <= hour < 7:
            labels.append("asia")
        elif 7 <= hour < 13:
            labels.append("europe")
        elif 13 <= hour < 20:
            labels.append("us")
        else:
            labels.append("after_hours")
    return np.array(labels)


def aggregate_metrics(predictions: pd.DataFrame, session_timezone: str = "UTC") -> pd.DataFrame:
    """Aggregate evaluation metrics across pairs, horizons and stratifications."""

    if not REQUIRED_COLUMNS.issubset(predictions.columns):
        missing = REQUIRED_COLUMNS.difference(predictions.columns)
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    frame = predictions.copy()
    frame["horizon"] = frame["horizon"].apply(_parse_horizon)
    frame["timestamp"] = _ensure_timezone(frame["timestamp"], "UTC")
    frame["session_ts"] = frame["timestamp"].dt.tz_convert(session_timezone)
    records: List[dict[str, object]] = []
    grouped = frame.groupby(["pair", "horizon"], sort=False)
    for (pair, horizon), group in grouped:
        metrics = _base_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        for metric_name, value in metrics.items():
            records.append(
                {
                    "pair": pair,
                    "horizon": horizon,
                    "group": "overall",
                    "segment": "all",
                    "metric": metric_name,
                    "value": value,
                    "count": len(group),
                }
            )
        if len(group) >= 3:
            regimes = _volatility_regime_labels(group["y_true"].to_numpy())
            for regime in np.unique(regimes):
                mask = regimes == regime
                regime_metrics = _base_metrics(
                    group.loc[mask, "y_true"].to_numpy(),
                    group.loc[mask, "y_pred"].to_numpy(),
                )
                for metric_name, value in regime_metrics.items():
                    records.append(
                        {
                            "pair": pair,
                            "horizon": horizon,
                            "group": "volatility",
                            "segment": regime,
                            "metric": metric_name,
                            "value": value,
                            "count": int(mask.sum()),
                        }
                    )
        sessions = _session_labels(group["session_ts"])
        for session in np.unique(sessions):
            mask = sessions == session
            session_metrics = _base_metrics(
                group.loc[mask, "y_true"].to_numpy(),
                group.loc[mask, "y_pred"].to_numpy(),
            )
            for metric_name, value in session_metrics.items():
                records.append(
                    {
                        "pair": pair,
                        "horizon": horizon,
                        "group": "session",
                        "segment": session,
                        "metric": metric_name,
                        "value": value,
                        "count": int(mask.sum()),
                    }
                )
    return pd.DataFrame.from_records(records)


def run_evaluation(
    predictions_path: Path,
    run_id: str,
    artifacts_dir: Path = Path("artifacts"),
    session_timezone: str = "UTC",
) -> Path:
    """Load predictions, aggregate metrics, and persist summaries to disk."""

    predictions = pd.read_csv(predictions_path)
    metrics = aggregate_metrics(predictions, session_timezone=session_timezone)
    output_dir = artifacts_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metrics.csv"
    metrics.to_csv(output_path, index=False)
    return output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate FX evaluation metrics")
    parser.add_argument("--run-id", required=True, help="Identifier for the evaluation run")
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Path to a CSV file containing pair/horizon predictions",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where aggregated metrics will be stored",
    )
    parser.add_argument(
        "--session-timezone",
        default="UTC",
        help="Timezone used for trading session bucketing",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    output_path = run_evaluation(
        predictions_path=args.predictions,
        run_id=args.run_id,
        artifacts_dir=args.artifacts_dir,
        session_timezone=args.session_timezone,
    )
    LOGGER.info("Saved aggregated metrics to %s", output_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    main()
