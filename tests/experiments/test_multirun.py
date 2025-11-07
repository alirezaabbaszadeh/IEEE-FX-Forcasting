"""Smoke tests for the multirun experiment utilities."""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.experiments.multirun import RunResult, run_multirun
from src.experiments.pcc import run_pcc_toggle
from src.experiments.runner import MultiRunExperiment
from src.experiments.runner import VariantRunCollection
from src.training.engine import ComputeStats


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
    compute: ComputeStats | None = None


def _make_summary(*, best: float, final_loss: float, final_mae: float) -> _Summary:
    compute = ComputeStats(
        wall_time_s=12.0,
        cpu_rss_mb_mean=512.0,
        cpu_rss_mb_peak=520.0,
        gpu_utilization_mean=float("nan"),
        gpu_utilization_max=float("nan"),
        gpu_memory_mb_peak=float("nan"),
        samples=4,
    )
    return _Summary(
        epochs=[_Epoch(train_loss=1.0, val_loss=final_loss, val_mae=final_mae)],
        best_val_loss=best,
        compute=compute,
    )


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
        assert "deterministic_flags" in metadata
        assert metadata.get("git_sha") == "abcdef"
        assert "hardware" in metadata
        assert metadata["artifacts"]["metrics"] == "metrics.json"
        assert metadata["artifacts"]["metadata"] == "metadata.json"
        assert metadata["artifacts"]["resolved_config"] == "resolved_config.yaml"
        assert metadata["artifacts"]["compute"] == "compute.json"
        assert metadata["artifacts"]["compute_csv"] == "compute.csv"
        assert "compute" in metadata

        resolved_path = seed_dir / "resolved_config.yaml"
        assert resolved_path.exists()
        assert resolved_path.read_text().strip() != ""

        manifest_path = seed_dir / "manifest.json"
        assert manifest_path.exists()
        manifest_payload = json.loads(manifest_path.read_text())
        assert manifest_payload.get("git")

        compute_payload = json.loads((seed_dir / "compute.json").read_text())
        assert compute_payload["seed"] == seed
        assert compute_payload["epochs"] == len(summaries[seed].epochs)
        compute_csv = seed_dir / "compute.csv"
        assert compute_csv.exists()
        csv_rows = list(csv.DictReader(compute_csv.open()))
        assert csv_rows
        assert "wall_time_s" in csv_rows[0]

    summary_path = tmp_path / "summary.csv"
    assert summary_path.exists()
    rows = {row["metric"]: row for row in csv.DictReader(summary_path.open())}
    best_metrics = rows["best_val_loss"]
    # mean should be the mid point and std non-zero for differing inputs
    assert abs(float(best_metrics["mean"]) - 0.7) < 1e-6
    assert float(best_metrics["std"]) > 0.0
    assert float(best_metrics["ci95"]) > 0.0
    assert float(best_metrics["n"]) == len(seeds)

    metadata_blob = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata_blob["seeds"] == seeds
    assert metadata_blob["metrics"]["final_val_mae"]["mean"] == aggregated["final_val_mae"]["mean"]
    assert metadata_blob["deterministic_flags"]["deterministic_algorithms"] is True
    assert len(metadata_blob["run_records"]) == len(seeds)
    for record in metadata_blob["run_records"]:
        assert "metrics" in record
        assert "metadata" in record
        assert record["metadata"].get("git_sha") == "abcdef"
    assert metadata_blob["artifacts"]["summary"] == "summary.csv"
    assert set(metadata_blob["artifacts"]["runs"]) == {f"seed-{seed}" for seed in seeds}
    assert metadata_blob["artifacts"]["compute"].endswith("compute.json")
    assert metadata_blob["artifacts"]["resolved_config"].endswith("resolved_config.yaml")

    aggregate_dir = tmp_path / "aggregates"
    assert (aggregate_dir / "aggregate.csv").exists()
    assert (aggregate_dir / "calibration.csv").exists()
    assert (aggregate_dir / "compute.csv").exists()
    assert (aggregate_dir / "dm_table.csv").exists()
    assert (aggregate_dir / "spa_table.csv").exists()
    assert (aggregate_dir / "mcs_table.csv").exists()


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

    # Student-t critical value for df=1 is ~12.706
    ci_radius = 12.706 * expected_std / len(seeds) ** 0.5
    assert abs(aggregated["best_val_loss"]["ci95"] - ci_radius) < 1e-6


def test_run_variants_writes_summary_and_enforces_threshold(tmp_path: Path) -> None:
    seeds = list(range(10, 20))

    def _variant_runner(config: dict[str, object]) -> dict[str, object]:
        seed = int(config["seed"])
        noise = (seed % 5) * 0.0005
        use_pcc = bool(config.get("use_pcc", False))
        base_crps = 0.55 - (0.05 if use_pcc else 0.0)
        base_cov = 0.10 - (0.04 if use_pcc else 0.0)
        return {
            "metrics": {
                "crps": base_crps + noise,
                "coverage_error": base_cov + noise,
            }
        }

    experiment = MultiRunExperiment(_variant_runner, num_runs=len(seeds), base_seed=min(seeds))

    collection = experiment.run_variants(
        {"use_pcc": False},
        run_id="pcc_case",
        variants={"pcc_off": {"use_pcc": False}, "pcc_on": {"use_pcc": True}},
        baseline="pcc_off",
        output_dir=tmp_path,
        seeds=seeds,
        improvement_thresholds={"crps": 0.02, "coverage_error": 0.02},
    )

    assert isinstance(collection, VariantRunCollection)
    assert set(collection.results) == {"pcc_off", "pcc_on"}

    summary_path = tmp_path / "pcc_case" / "variants.json"
    assert collection.summary_path == summary_path
    assert summary_path.exists()

    payload = json.loads(summary_path.read_text())
    thresholds = payload.get("thresholds", {})
    assert pytest.approx(thresholds["crps"]["relative"]) == 0.02

    variant_payload = {entry["name"]: entry for entry in payload["variants"]}
    pcc_on = variant_payload["pcc_on"]
    deltas = pcc_on["deltas"]
    assert deltas["crps"]["relative"] > 0.02
    assert deltas["coverage_error"]["relative"] > 0.02


def test_run_variants_raises_when_improvement_below_threshold(tmp_path: Path) -> None:
    seeds = list(range(7, 17))

    def _variant_runner(config: dict[str, object]) -> dict[str, object]:
        seed = int(config["seed"])
        noise = (seed % 3) * 0.0002
        use_pcc = bool(config.get("use_pcc", False))
        base_crps = 0.5 - (0.009 if use_pcc else 0.0)
        base_cov = 0.08 - (0.025 if use_pcc else 0.0)
        return {
            "metrics": {
                "crps": base_crps + noise,
                "coverage_error": base_cov + noise,
            }
        }

    experiment = MultiRunExperiment(_variant_runner, num_runs=len(seeds), base_seed=min(seeds))

    with pytest.raises(ValueError):
        experiment.run_variants(
            {"use_pcc": False},
            run_id="pcc_fail",
            variants={"pcc_off": {"use_pcc": False}, "pcc_on": {"use_pcc": True}},
            baseline="pcc_off",
            output_dir=tmp_path,
            seeds=seeds,
            improvement_thresholds={"crps": 0.02, "coverage_error": 0.02},
        )


def test_run_pcc_toggle_executes_grid(tmp_path: Path) -> None:
    seeds = list(range(50, 60))

    def _runner(config: dict[str, object]) -> dict[str, object]:
        seed = int(config["seed"])
        noise = (seed % 4) * 0.0003
        use_pcc = bool(config.get("use_pcc", False))
        base_crps = 0.6 - (0.06 if use_pcc else 0.0)
        base_cov = 0.12 - (0.05 if use_pcc else 0.0)
        return {
            "metrics": {
                "crps": base_crps + noise,
                "coverage_error": base_cov + noise,
            }
        }

    results = run_pcc_toggle(
        _runner,
        run_id="pcc_ablation",
        pairs=["EURUSD", "GBPUSD", "USDJPY"],
        horizons=["1h", "4h"],
        seeds=seeds,
        output_dir=tmp_path,
        base_config={"other_param": 1},
        variant_field="use_pcc",
        improvement_thresholds={"crps": {"relative": 0.02}, "coverage_error": {"relative": 0.02}},
    )

    assert len(results) == 6
    sample = results[("EURUSD", "1h")]
    assert isinstance(sample, VariantRunCollection)
    assert set(sample.results) == {"pcc_off", "pcc_on"}

    manifest_path = tmp_path / "pcc_ablation" / "pcc_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == "pcc_ablation"
    assert len(manifest["entries"]) == 6
