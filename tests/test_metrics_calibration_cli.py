from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics.calibration_cli import generate_calibration


def _write_predictions(path: Path, seed: int, n: int = 128) -> Path:
    rng = np.random.default_rng(seed)
    y_true = rng.normal(loc=0.0, scale=1.0, size=n)
    samples = rng.normal(loc=y_true[:, None], scale=0.5, size=(n, 50))

    frame = pd.DataFrame({"y_true": y_true})
    for idx in range(samples.shape[1]):
        frame[f"sample_{idx}"] = samples[:, idx]

    for level in (0.1, 0.5, 0.9):
        quantile = np.quantile(samples, level, axis=1)
        frame[f"quantile_{level:.2f}"] = quantile

    output_path = path / "predictions.csv"
    frame.to_csv(output_path, index=False)
    return output_path


def test_generate_calibration_writes_expected_outputs(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    file_a = _write_predictions(run_a, seed=3)
    file_b = _write_predictions(run_b, seed=5)

    output_root = tmp_path / "paper_outputs"
    result_path = generate_calibration(
        [file_a, file_b],
        output_root=output_root,
        quantiles=(0.1, 0.5, 0.9),
        intervals=(0.5, 0.8),
        pit_bins=8,
    )

    assert result_path.exists()
    summary = pd.read_csv(result_path)
    assert not summary.empty
    assert {"run", "metric", "level", "value"}.issubset(summary.columns)
    assert ((summary["run"] == "overall") & (summary["metric"] == "crps")).any()

    figs_dir = output_root / "figs"
    expected_labels = {"run_a_predictions", "run_b_predictions", "overall"}
    for label in expected_labels:
        pit_path = figs_dir / f"pit_hist_{label}.png"
        coverage_path = figs_dir / f"coverage_{label}.png"
        assert pit_path.exists()
        assert pit_path.stat().st_size > 0
        assert coverage_path.exists()
        assert coverage_path.stat().st_size > 0

    coverage_rows = summary[summary["metric"] == "coverage"]
    assert not coverage_rows.empty
    assert coverage_rows["value"].between(0.0, 1.0).all()
