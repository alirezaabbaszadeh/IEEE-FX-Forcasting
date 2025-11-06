"""Tests for reporting aggregation utilities."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from src.experiments.multirun import RunResult, run_multirun
from src.reporting.aggregates import collate_run_group


@dataclass
class _Epoch:
    train_loss: float
    val_loss: float
    val_mae: float


@dataclass
class _Summary:
    epochs: list[_Epoch]
    best_val_loss: float
    device: str = "cpu"


def _build_summary(best: float, final_loss: float, final_mae: float) -> _Summary:
    return _Summary(epochs=[_Epoch(train_loss=1.0, val_loss=final_loss, val_mae=final_mae)], best_val_loss=best)


def test_collate_run_group_emits_expected_tables(tmp_path: Path) -> None:
    seeds = [3, 5]
    base_metadata = {
        "config_hash": "abc123",
        "git_sha": "deadbeef",
        "hardware": {"cpu": "fixture"},
        "dataset": {"pair": "EURUSD", "horizon": 1, "window_id": 0},
    }

    summaries = {
        3: _build_summary(best=0.9, final_loss=0.91, final_mae=0.42),
        5: _build_summary(best=0.7, final_loss=0.73, final_mae=0.38),
    }

    run_root = tmp_path / "artifacts" / "runs" / "demo_model" / "abc123" / "eurusd_1" / "window-000"

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

    aggregated = run_multirun(seeds, run_root, _runner, base_metadata=base_metadata)
    assert aggregated

    outputs = collate_run_group(run_root)
    expected_keys = {"aggregate", "calibration", "dm_table", "spa_table", "mcs_table"}
    assert expected_keys.issubset(outputs.keys())

    aggregates_dir = tmp_path / "artifacts" / "aggregates" / "demo_model" / "abc123" / "eurusd_1" / "window-000"
    for key in expected_keys:
        path = outputs[key]
        assert path.exists()
        assert path.is_file()
        assert path.read_text().strip() != ""
        assert path.parent == aggregates_dir

    with (aggregates_dir / "aggregate.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["metric"]

    compute_path = run_root / "seed-3" / "compute.json"
    assert compute_path.exists()
    compute_payload = json.loads(compute_path.read_text())
    assert compute_payload["seed"] == 3
