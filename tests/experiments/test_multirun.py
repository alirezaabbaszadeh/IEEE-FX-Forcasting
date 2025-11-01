"""Smoke tests for the multirun experiment utilities."""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.experiments.multirun import RunResult, run_multirun


@dataclass
class _Epoch:
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_mae: float = 0.0


@dataclass
class _Summary:
    epochs: List[_Epoch]
    best_val_loss: float
    device: str = "cpu"


def _make_summary(*, best: float, final_loss: float, final_mae: float) -> _Summary:
    return _Summary(epochs=[_Epoch(train_loss=1.0, val_loss=final_loss, val_mae=final_mae)], best_val_loss=best)


def test_run_multirun_writes_expected_layout(tmp_path: Path) -> None:
    seeds = [7, 11]
    base_metadata = {"config_hash": "hash", "git_sha": "abcdef", "hardware": {"cpu": "test"}}

    summaries = {
        7: _make_summary(best=0.8, final_loss=0.82, final_mae=0.41),
        11: _make_summary(best=0.6, final_loss=0.64, final_mae=0.39),
    }

    def _runner(seed: int) -> RunResult:
        return RunResult(
            seed=seed,
            summary=summaries[seed],
            metadata={
                **base_metadata,
                "device": "cpu",
                "deterministic_flags": {"deterministic_algorithms": True},
            },
        )

    aggregated = run_multirun(seeds, tmp_path, _runner, base_metadata=base_metadata)

    for seed in seeds:
        seed_dir = tmp_path / f"seed-{seed}"
        assert seed_dir.exists()
        metrics = json.loads((seed_dir / "metrics.json").read_text())
        metadata = json.loads((seed_dir / "metadata.json").read_text())
        assert metrics["best_val_loss"] == summaries[seed].best_val_loss
        assert metadata["seed"] == seed
        assert metadata["device"] == "cpu"
        assert metadata["config_hash"] == "hash"

    summary_path = tmp_path / "summary.csv"
    assert summary_path.exists()
    rows = {row["metric"]: row for row in csv.DictReader(summary_path.open())}
    best_metrics = rows["best_val_loss"]
    # mean should be the mid point and std non-zero for differing inputs
    assert abs(float(best_metrics["mean"]) - 0.7) < 1e-6
    assert float(best_metrics["std"]) > 0.0
    assert float(best_metrics["ci95"]) > 0.0

    metadata_blob = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata_blob["seeds"] == seeds
    assert metadata_blob["metrics"]["final_val_mae"]["mean"] == aggregated["final_val_mae"]["mean"]
    assert metadata_blob["deterministic_flags"] == {"deterministic_algorithms": True}


def test_run_multirun_aggregation_math(tmp_path: Path) -> None:
    seeds = [1, 2]
    base_metadata = {"config_hash": "xyz", "git_sha": "012345", "hardware": {}}

    summaries = {
        1: _make_summary(best=1.0, final_loss=1.2, final_mae=0.5),
        2: _make_summary(best=1.4, final_loss=1.6, final_mae=0.7),
    }

    def _runner(seed: int) -> RunResult:
        return RunResult(
            seed=seed,
            summary=summaries[seed],
            metadata={**base_metadata, "device": "cpu"},
        )

    aggregated = run_multirun(seeds, tmp_path, _runner, base_metadata=base_metadata)

    expected_mean = (1.0 + 1.4) / 2
    assert abs(aggregated["best_val_loss"]["mean"] - expected_mean) < 1e-6

    # Sample standard deviation for values (1.0, 1.4)
    expected_std = (
        ((1.0 - expected_mean) ** 2 + (1.4 - expected_mean) ** 2) / (len(seeds) - 1)
    ) ** 0.5
    assert abs(aggregated["best_val_loss"]["std"] - expected_std) < 1e-6

    ci_radius = 1.96 * expected_std / len(seeds) ** 0.5
    assert abs(aggregated["best_val_loss"]["ci95"] - ci_radius) < 1e-6
