from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.analysis import StatisticalAnalyzer
from src.experiments import HyperparameterSearch, MultiRunExperiment


def _dummy_run(config):
    seed = config["seed"]
    base = config.get("base", 0.0)
    rng = np.random.default_rng(seed)
    value = float(base + rng.normal(scale=0.01))
    return {"metrics": {"val_score": value}, "raw": value}


def test_multi_run_reproducibility(tmp_path: Path) -> None:
    runner = MultiRunExperiment(_dummy_run, num_runs=5, base_seed=123)
    config = {"base": 0.5}
    result_a = runner.run(config, run_id="exp_a", output_dir=tmp_path)
    result_b = runner.run(config, run_id="exp_b", output_dir=tmp_path)

    assert result_a.aggregate == result_b.aggregate

    metadata_path = tmp_path / "exp_a" / "metadata.json"
    assert metadata_path.exists()
    payload = json.loads(metadata_path.read_text())
    assert payload["aggregate"]["val_score"]["n"] == 5
    assert "ci95_low" in payload["aggregate"]["val_score"]
    assert len(payload["runs"]) == 5
    for run in payload["runs"]:
        assert "metrics" in run
        assert "extra" in run


def test_statistical_analysis_outputs(tmp_path: Path) -> None:
    metrics = {
        "baseline": [0.55, 0.53, 0.51, 0.54, 0.52],
        "model_b": [0.50, 0.48, 0.52, 0.49, 0.51],
        "model_c": [0.60, 0.58, 0.62, 0.59, 0.61],
    }
    analyzer = StatisticalAnalyzer(run_id="analysis", output_dir=tmp_path)
    tables = analyzer.analyze(metrics, baseline="baseline")

    stats_dir = tmp_path / "analysis" / "stats"
    expected_files = {
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
    assert expected_files == {path.stem for path in stats_dir.iterdir() if path.suffix == ".csv"}

    effect_sizes = tables["effect_sizes"]
    assert set(effect_sizes.columns) >= {"cohens_d", "hedges_g", "glass_delta", "rank_biserial"}
    dm_table = tables["diebold_mariano"]
    assert {"dm_stat", "variance"}.issubset(dm_table.columns)

    bootstrap = tables["bootstrap_ci"]
    assert {"estimate", "lower", "upper", "coverage"}.issubset(bootstrap.columns)


def test_hyperparameter_search_stub(tmp_path: Path) -> None:
    runner = MultiRunExperiment(_dummy_run, num_runs=5, base_seed=0)
    base_config = {"base": 0.5}
    search_space = {
        "base": {"type": "float", "low": 0.4, "high": 0.6},
    }

    search = HyperparameterSearch(
        runner=runner,
        base_config=base_config,
        search_space=search_space,
        metric="val_score",
        run_id="search_case",
        output_dir=tmp_path,
        direction="maximize",
    )

    result = search.optimize(n_trials=3, sampler="sobol")
    assert "best_value" in result
    assert (tmp_path / "search_case" / "search_results.csv").exists()

