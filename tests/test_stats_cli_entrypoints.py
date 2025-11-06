from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.stats import analyze_dm_cache
from src.stats.pbo import probability_of_backtest_overfitting


@pytest.fixture
def dm_cache_fixture() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=6, freq="H", tz="UTC")
    baseline_pred = [0.1, -0.05, 0.02, -0.01, 0.03, -0.02]
    alt_pred = [val + adj for val, adj in zip(baseline_pred, [0.01, -0.02, 0.015, -0.005, 0.02, -0.01], strict=True)]
    challenger_pred = [val + adj for val, adj in zip(baseline_pred, [-0.015, 0.01, -0.005, 0.02, -0.01, 0.015], strict=True)]
    truth = [0.0, -0.02, 0.01, -0.01, 0.02, -0.015]

    rows = []
    for model, preds in {
        "baseline": baseline_pred,
        "alt": alt_pred,
        "challenger": challenger_pred,
    }.items():
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


def _write_dm_cache(cache: pd.DataFrame, path: Path) -> Path:
    path.write_text(cache.to_csv(index=False))
    return path


def _analysis_tables(cache: pd.DataFrame, tmp_path: Path, n_bootstrap: int) -> dict[str, pd.DataFrame]:
    return analyze_dm_cache(
        cache,
        run_id="paper",
        output_dir=tmp_path / "analysis",
        baseline_model="baseline",
        metric="squared_error",
        alpha=0.1,
        assumption_alpha=0.1,
        newey_west_lag=1,
        higher_is_better=False,
        random_state=123,
        n_bootstrap=n_bootstrap,
    )


def test_dm_cli_matches_expected_table(dm_cache_fixture: pd.DataFrame, tmp_path: Path) -> None:
    cache_path = _write_dm_cache(dm_cache_fixture, tmp_path / "dm_cache.csv")
    output_dir = tmp_path / "dm_outputs"
    n_bootstrap = 256
    cmd = [
        sys.executable,
        "-m",
        "src.stats.dm",
        str(cache_path),
        "--baseline-model",
        "baseline",
        "--metric",
        "squared_error",
        "--run-id",
        "paper",
        "--output-dir",
        str(output_dir),
        "--alpha",
        "0.1",
        "--assumption-alpha",
        "0.1",
        "--newey-west-lag",
        "1",
        "--random-seed",
        "123",
        "--n-bootstrap",
        str(n_bootstrap),
    ]
    subprocess.run(cmd, check=True)

    produced = pd.read_csv(output_dir / "diebold_mariano.csv").sort_values(
        ["pair", "horizon", "model_a", "model_b"]
    ).reset_index(drop=True)
    expected = (
        _analysis_tables(dm_cache_fixture, tmp_path, n_bootstrap)["diebold_mariano"]
        .sort_values(["pair", "horizon", "model_a", "model_b"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(produced, expected)


def test_spa_cli_emits_bootstrap_p_values(dm_cache_fixture: pd.DataFrame, tmp_path: Path) -> None:
    cache_path = _write_dm_cache(dm_cache_fixture, tmp_path / "dm_cache.csv")
    output_dir = tmp_path / "spa_outputs"
    n_bootstrap = 256
    cmd = [
        sys.executable,
        "-m",
        "src.stats.spa",
        str(cache_path),
        "--baseline-model",
        "baseline",
        "--metric",
        "squared_error",
        "--run-id",
        "paper",
        "--output-dir",
        str(output_dir),
        "--alpha",
        "0.1",
        "--assumption-alpha",
        "0.1",
        "--newey-west-lag",
        "1",
        "--random-seed",
        "123",
        "--n-bootstrap",
        str(n_bootstrap),
    ]
    subprocess.run(cmd, check=True)

    produced = pd.read_csv(output_dir / "spa.csv").sort_values(["pair", "horizon", "model"]).reset_index(drop=True)
    expected = (
        _analysis_tables(dm_cache_fixture, tmp_path, n_bootstrap)["model_confidence_set"]
        [["pair", "horizon", "model", "spa_p_value", "ci_lower", "ci_upper"]]
        .sort_values(["pair", "horizon", "model"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(produced, expected)


def test_mcs_cli_matches_confidence_set(dm_cache_fixture: pd.DataFrame, tmp_path: Path) -> None:
    cache_path = _write_dm_cache(dm_cache_fixture, tmp_path / "dm_cache.csv")
    output_dir = tmp_path / "mcs_outputs"
    n_bootstrap = 256
    cmd = [
        sys.executable,
        "-m",
        "src.stats.mcs",
        str(cache_path),
        "--baseline-model",
        "baseline",
        "--metric",
        "squared_error",
        "--run-id",
        "paper",
        "--output-dir",
        str(output_dir),
        "--alpha",
        "0.1",
        "--assumption-alpha",
        "0.1",
        "--newey-west-lag",
        "1",
        "--random-seed",
        "123",
        "--n-bootstrap",
        str(n_bootstrap),
    ]
    subprocess.run(cmd, check=True)

    produced = pd.read_csv(output_dir / "model_confidence_set.csv").sort_values(
        ["pair", "horizon", "model"]
    ).reset_index(drop=True)
    expected = (
        _analysis_tables(dm_cache_fixture, tmp_path, n_bootstrap)["model_confidence_set"]
        .sort_values(["pair", "horizon", "model"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(produced, expected)


def test_pbo_cli_matches_probability_estimate(dm_cache_fixture: pd.DataFrame, tmp_path: Path) -> None:
    cache_path = _write_dm_cache(dm_cache_fixture, tmp_path / "dm_cache.csv")
    output_dir = tmp_path / "pbo_outputs"
    n_bootstrap = 64
    max_combinations = 16
    cmd = [
        sys.executable,
        "-m",
        "src.stats.pbo",
        str(cache_path),
        "--baseline-model",
        "baseline",
        "--metric",
        "squared_error",
        "--run-id",
        "paper",
        "--output-dir",
        str(output_dir),
        "--alpha",
        "0.1",
        "--assumption-alpha",
        "0.1",
        "--newey-west-lag",
        "1",
        "--random-seed",
        "123",
        "--max-combinations",
        str(max_combinations),
        "--n-bootstrap",
        str(n_bootstrap),
    ]
    subprocess.run(cmd, check=True)

    produced = pd.read_csv(output_dir / "pbo.csv").sort_values(["pair", "horizon"]).reset_index(drop=True)
    expected = probability_of_backtest_overfitting(
        dm_cache_fixture,
        metric="squared_error",
        higher_is_better=False,
        max_combinations=max_combinations,
        random_state=123,
    ).sort_values(["pair", "horizon"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(produced, expected)
