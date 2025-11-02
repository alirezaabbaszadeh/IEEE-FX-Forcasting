from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.stats import analyze_dm_cache


def _build_dm_cache() -> pd.DataFrame:
    timestamps = pd.date_range("2023-01-01", periods=4, freq="H", tz="UTC")
    baseline_pred = [0.1, -0.1, 0.05, -0.02]
    alt_pred = [val + adj for val, adj in zip(baseline_pred, [0.02, -0.03, 0.01, -0.01], strict=True)]
    truth = [0.0, -0.05, 0.02, 0.0]

    rows = []
    for model, preds in {"baseline": baseline_pred, "alt": alt_pred}.items():
        for ts, y_true, y_pred in zip(timestamps, truth, preds, strict=True):
            error = float(y_pred - y_true)
            rows.append(
                {
                    "pair": "EURUSD",
                    "horizon": "1h",
                    "model": model,
                    "timestamp": ts.isoformat(),
                    "timestamp_utc": ts.isoformat(),
                    "y_true": float(y_true),
                    "y_pred": float(y_pred),
                    "error": error,
                    "abs_error": float(abs(error)),
                    "squared_error": float(error**2),
                }
            )
    return pd.DataFrame(rows)


def test_analyze_dm_cache_creates_stat_tables(tmp_path: Path) -> None:
    dm_cache = _build_dm_cache()
    tables = analyze_dm_cache(
        dm_cache,
        run_id="evaluation",
        output_dir=tmp_path,
        baseline_model="baseline",
        metric="squared_error",
        alpha=0.1,
        assumption_alpha=0.1,
        newey_west_lag=1,
        higher_is_better=False,
    )

    stats_root = tmp_path / "evaluation" / "stats"
    assert stats_root.exists()

    expected = {
        "anova",
        "welch",
        "kruskal",
        "effect_sizes",
        "diebold_mariano",
        "bootstrap_ci",
        "posthoc_tukey",
        "posthoc_dunn",
        "model_confidence_set",
    }
    produced = {path.stem for path in stats_root.glob("*.csv")}
    assert expected == produced

    effect_sizes = tables.get("effect_sizes")
    assert effect_sizes is not None
    assert not effect_sizes.empty

    dm_table = tables.get("diebold_mariano")
    assert dm_table is not None
    assert {"model_a", "model_b"}.issubset(dm_table.columns)
