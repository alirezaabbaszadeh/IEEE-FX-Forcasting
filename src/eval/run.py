
"""Evaluation runner that aggregates metrics across pairs and horizons."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from src.analysis.stats import analyze_dm_cache
from src.metrics.point import point_metrics

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"pair", "horizon", "timestamp", "y_true", "y_pred"}
DEFAULT_MODEL_NAME = "model_0"
DM_CACHE_COLUMNS = (
    "pair",
    "horizon",
    "model",
    "timestamp",
    "timestamp_utc",
    "y_true",
    "y_pred",
    "error",
    "abs_error",
    "squared_error",
)


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


def _compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = point_metrics(y_true, y_pred)
    metrics["directional_accuracy"] = _directional_accuracy(y_true, y_pred)
    return metrics


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


def _compute_gating_entropy(frame: pd.DataFrame) -> np.ndarray | None:
    gating_cols = [col for col in frame.columns if col.startswith("gate_prob_")]
    if gating_cols:
        probs = frame[gating_cols].to_numpy(dtype=float)
        if probs.ndim != 2:  # pragma: no cover - sanity check
            raise ValueError("Gate probability columns must form a 2D array")
        sums = probs.sum(axis=1, keepdims=True)
        sums[sums == 0.0] = 1.0
        normalised = probs / sums
        normalised = np.clip(normalised, 1e-8, 1.0)
        return -np.sum(normalised * np.log(normalised), axis=1)
    if "gating_entropy" in frame.columns:
        return frame["gating_entropy"].to_numpy(dtype=float)
    return None


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    corr = np.corrcoef(x, y)
    return float(corr[0, 1])


def aggregate_metrics(predictions: pd.DataFrame, session_timezone: str = "UTC") -> pd.DataFrame:
    """Aggregate evaluation metrics across pairs, horizons and stratifications."""

    if not REQUIRED_COLUMNS.issubset(predictions.columns):
        missing = REQUIRED_COLUMNS.difference(predictions.columns)
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    frame = predictions.copy()
    if "model" not in frame.columns:
        frame["model"] = DEFAULT_MODEL_NAME
    else:
        frame["model"] = frame["model"].fillna(DEFAULT_MODEL_NAME).astype(str)
    frame["horizon"] = frame["horizon"].apply(_parse_horizon)
    frame["timestamp"] = _ensure_timezone(frame["timestamp"], "UTC")
    frame["session_ts"] = frame["timestamp"].dt.tz_convert(session_timezone)
    records: List[dict[str, object]] = []
    dm_cache_frames: List[pd.DataFrame] = []
    grouped = frame.groupby(["pair", "horizon", "model"], sort=False)
    for (pair, horizon, model), group in grouped:
        y_true = group["y_true"].to_numpy()
        y_pred = group["y_pred"].to_numpy()
        metrics = _compute_point_metrics(y_true, y_pred)
        for metric_name, value in metrics.items():
            records.append(
                {
                    "pair": pair,
                    "horizon": horizon,
                    "model": model,
                    "group": "overall",
                    "segment": "all",
                    "metric": metric_name,
                    "value": value,
                    "count": len(group),
                }
            )
        if "fold" in group.columns:
            for fold_value, fold_frame in group.groupby("fold", sort=False):
                fold_metrics = _compute_point_metrics(
                    fold_frame["y_true"].to_numpy(), fold_frame["y_pred"].to_numpy()
                )
                for metric_name, value in fold_metrics.items():
                    records.append(
                        {
                            "pair": pair,
                            "horizon": horizon,
                            "model": model,
                            "group": "fold",
                            "segment": str(fold_value),
                            "metric": metric_name,
                            "value": value,
                            "count": len(fold_frame),
                        }
                    )
        gating_entropy = _compute_gating_entropy(group)
        if gating_entropy is not None and gating_entropy.size:
            volatility = np.abs(y_true)
            entropy_mean = float(np.mean(gating_entropy))
            entropy_corr = _safe_correlation(gating_entropy, volatility)
            LOGGER.info(
                "Pair %s | horizon %s | model %s - gating entropy mean: %.4f | corr(|y_true|): %s",
                pair,
                horizon,
                model,
                entropy_mean,
                "nan" if np.isnan(entropy_corr) else f"{entropy_corr:.4f}",
            )
            records.append(
                {
                    "pair": pair,
                    "horizon": horizon,
                    "model": model,
                    "group": "interpretability",
                    "segment": "gating",
                    "metric": "entropy_mean",
                    "value": entropy_mean,
                    "count": len(group),
                }
            )
            records.append(
                {
                    "pair": pair,
                    "horizon": horizon,
                    "model": model,
                    "group": "interpretability",
                    "segment": "gating",
                    "metric": "entropy_volatility_corr",
                    "value": float(entropy_corr),
                    "count": len(group),
                }
            )
        if len(group) >= 3:
            regimes = _volatility_regime_labels(group["y_true"].to_numpy())
            for regime in np.unique(regimes):
                mask = regimes == regime
                regime_metrics = _compute_point_metrics(
                    group.loc[mask, "y_true"].to_numpy(),
                    group.loc[mask, "y_pred"].to_numpy(),
                )
                for metric_name, value in regime_metrics.items():
                    records.append(
                        {
                            "pair": pair,
                            "horizon": horizon,
                            "model": model,
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
            session_metrics = _compute_point_metrics(
                group.loc[mask, "y_true"].to_numpy(),
                group.loc[mask, "y_pred"].to_numpy(),
            )
            for metric_name, value in session_metrics.items():
                records.append(
                    {
                        "pair": pair,
                        "horizon": horizon,
                        "model": model,
                        "group": "session",
                        "segment": session,
                        "metric": metric_name,
                        "value": value,
                        "count": int(mask.sum()),
                    }
                )
        errors = y_pred - y_true
        dm_cache_frames.append(
            pd.DataFrame(
                {
                    "pair": pair,
                    "horizon": horizon,
                    "model": model,
                    "timestamp": group["session_ts"].astype(str).to_numpy(),
                    "timestamp_utc": group["timestamp"].astype(str).to_numpy(),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "error": errors,
                    "abs_error": np.abs(errors),
                    "squared_error": np.square(errors),
                }
            )
        )
    metrics_df = pd.DataFrame.from_records(records)
    if dm_cache_frames:
        metrics_df.attrs["dm_cache"] = pd.concat(dm_cache_frames, ignore_index=True)[
            list(DM_CACHE_COLUMNS)
        ]
    else:
        metrics_df.attrs["dm_cache"] = pd.DataFrame(columns=DM_CACHE_COLUMNS)
    return metrics_df


def run_evaluation(
    predictions_path: Path,
    run_id: str,
    artifacts_dir: Path = Path("artifacts"),
    session_timezone: str = "UTC",
    baseline_model: str | None = None,
    alpha: float = 0.05,
    assumption_alpha: float = 0.05,
    newey_west_lag: int = 1,
    higher_is_better: bool = False,
    stats_metric: str = "squared_error",
) -> Path:
    """Load predictions, aggregate metrics, and persist summaries to disk."""

    predictions = pd.read_csv(predictions_path)
    metrics = aggregate_metrics(predictions, session_timezone=session_timezone)
    output_dir = artifacts_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metrics.csv"
    metrics.to_csv(output_path, index=False)

    dm_cache = metrics.attrs.get("dm_cache")
    dm_cache_path: Path | None = None
    if isinstance(dm_cache, pd.DataFrame) and not dm_cache.empty:
        dm_cache_path = output_dir / "dm_cache.csv"
        dm_cache.to_csv(dm_cache_path, index=False)
        LOGGER.info("Persisted DM cache to %s", dm_cache_path)
    elif isinstance(dm_cache, pd.DataFrame):
        LOGGER.warning("DM cache is empty; statistical tests will be skipped unless additional data is provided")

    if baseline_model and isinstance(dm_cache, pd.DataFrame) and not dm_cache.empty:
        stats_tables = analyze_dm_cache(
            dm_cache,
            run_id=run_id,
            output_dir=artifacts_dir,
            baseline_model=baseline_model,
            metric=stats_metric,
            alpha=alpha,
            assumption_alpha=assumption_alpha,
            newey_west_lag=newey_west_lag,
            higher_is_better=higher_is_better,
        )
        if stats_tables:
            stats_dir = artifacts_dir / run_id / "stats"
            LOGGER.info("Statistical summaries stored in %s", stats_dir)
            effect_sizes = stats_tables.get("effect_sizes")
            if effect_sizes is not None and not effect_sizes.empty:
                top_effect = effect_sizes.reindex(
                    effect_sizes["hedges_g"].abs().sort_values(ascending=False).index
                ).iloc[0]
                LOGGER.info(
                    "Largest effect vs baseline %s: %s | Hedges' g=%.4f | rank-biserial=%.4f",
                    baseline_model,
                    top_effect["comparison"],
                    float(top_effect.get("hedges_g", float("nan"))),
                    float(top_effect.get("rank_biserial", float("nan"))),
                )
        else:
            LOGGER.warning("Statistical analysis returned no tables; check DM cache contents")
    elif baseline_model:
        LOGGER.warning("Baseline model specified but DM cache unavailable; skipping statistical analysis")

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
    parser.add_argument(
        "--baseline-model",
        help="Model name to treat as the baseline for effect sizes and Tukey HSD",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for omnibus and post-hoc tests",
    )
    parser.add_argument(
        "--assumption-alpha",
        type=float,
        default=0.05,
        help="Significance level used for normality and homoscedasticity diagnostics",
    )
    parser.add_argument(
        "--newey-west-lag",
        type=int,
        default=1,
        help="Lag parameter for Newey-West variance estimation in DM tests",
    )
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Treat the metric as higher-is-better when computing effect sizes",
    )
    parser.add_argument(
        "--stats-metric",
        default="squared_error",
        choices=["error", "abs_error", "squared_error"],
        help="Loss metric from the DM cache to analyse",
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
        baseline_model=args.baseline_model,
        alpha=args.alpha,
        assumption_alpha=args.assumption_alpha,
        newey_west_lag=args.newey_west_lag,
        higher_is_better=args.higher_is_better,
        stats_metric=args.stats_metric,
    )
    LOGGER.info("Saved aggregated metrics to %s", output_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    main()
