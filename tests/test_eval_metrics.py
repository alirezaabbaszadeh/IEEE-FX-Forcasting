from __future__ import annotations

import pandas as pd
import pytest

from src.eval.run import DM_CACHE_COLUMNS, aggregate_metrics


def test_aggregate_metrics_includes_gating_statistics(caplog):
    frame = pd.DataFrame(
        {
            "pair": ["EURUSD"] * 4,
            "horizon": ["1h"] * 4,
            "timestamp": [
                "2023-01-01T00:00:00Z",
                "2023-01-01T01:00:00Z",
                "2023-01-01T02:00:00Z",
                "2023-01-01T03:00:00Z",
            ],
            "y_true": [0.1, -0.2, 0.05, -0.1],
            "y_pred": [0.12, -0.18, 0.02, -0.12],
            "gate_prob_0": [0.6, 0.7, 0.8, 0.65],
            "gate_prob_1": [0.4, 0.3, 0.2, 0.35],
        }
    )

    with caplog.at_level("INFO"):
        metrics = aggregate_metrics(frame)

    gating_rows = metrics[(metrics["group"] == "interpretability") & (metrics["metric"] == "entropy_mean")]
    assert not gating_rows.empty
    assert gating_rows.iloc[0]["value"] > 0

    corr_rows = metrics[(metrics["group"] == "interpretability") & (metrics["metric"] == "entropy_volatility_corr")]
    assert not corr_rows.empty
    assert "gating entropy mean" in caplog.text


def test_aggregate_metrics_dm_cache_structure():
    frame = pd.DataFrame(
        {
            "pair": ["EURUSD"] * 3 + ["EURUSD"] * 3,
            "horizon": ["1h"] * 6,
            "model": ["baseline"] * 3 + ["alt"] * 3,
            "timestamp": [
                "2023-01-01T00:00:00Z",
                "2023-01-01T01:00:00Z",
                "2023-01-01T02:00:00Z",
                "2023-01-01T00:00:00Z",
                "2023-01-01T01:00:00Z",
                "2023-01-01T02:00:00Z",
            ],
            "y_true": [0.1, -0.2, 0.0, 0.1, -0.2, 0.0],
            "y_pred": [0.11, -0.21, 0.02, 0.13, -0.25, 0.05],
        }
    )

    metrics = aggregate_metrics(frame)
    assert "model" in metrics.columns

    dm_cache = metrics.attrs.get("dm_cache")
    assert isinstance(dm_cache, pd.DataFrame)
    assert not dm_cache.empty
    assert set(DM_CACHE_COLUMNS).issubset(dm_cache.columns)

    sample_row = dm_cache.iloc[0]
    assert abs(sample_row["error"]) == pytest.approx(abs(sample_row["y_pred"] - sample_row["y_true"]))
    assert sample_row["squared_error"] == pytest.approx(sample_row["error"] ** 2)
