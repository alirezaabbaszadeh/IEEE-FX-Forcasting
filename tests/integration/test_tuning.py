from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments.tuning import HyperparameterTuner, SearchSpaceConfig
from src.models.forecasting import ModelConfig
from src.training.engine import TrainerConfig


def _evaluation_fn(
    model_cfg: ModelConfig,
    trainer_cfg: TrainerConfig,
    seed: int,
    window_weight: float,
) -> dict[str, float]:
    rng = np.random.default_rng(seed + int(window_weight * 100))
    signal = window_weight + 0.02 * (model_cfg.hidden_size / 64) - trainer_cfg.learning_rate * 10
    noise = rng.normal(scale=0.001)
    return {"val_score": float(signal + noise)}


def test_tuning_ranking_outputs_are_deterministic(tmp_path: Path) -> None:
    search_yaml = tmp_path / "search.yaml"
    search_yaml.write_text(
        """
model:
  hidden_size:
    type: categorical
    choices: [64, 128]
training:
  learning_rate:
    type: categorical
    choices: [0.001, 0.0005]
evaluation:
  metric: val_score
  direction: maximize
  seeds: [11, 13, 17, 19, 23]
  top_k: 2
  sampler_seed: 7
        """.strip()
    )

    base_model = ModelConfig(input_features=4, time_steps=16)
    base_trainer = TrainerConfig()

    windows = {"w0": 0.1, "w1": 0.3}

    tuner = HyperparameterTuner.from_yaml(
        search_yaml,
        _evaluation_fn,
        base_model_cfg=base_model,
        base_trainer_cfg=base_trainer,
        windows=windows,
        run_id="test_run",
        output_dir=tmp_path / "artifacts",
    )

    result = tuner.optimize(n_trials=4)

    results_df = pd.read_csv(result["results_path"])
    ranking_df = pd.read_csv(result["ranking_path"])

    for plot_path in result["plot_paths"]:
        assert Path(plot_path).exists()

    # Reconstruct expected per-window means
    expected_records = []
    for trial_row in results_df.to_dict("records"):
        params = {
            "hidden_size": trial_row["model.hidden_size"],
            "learning_rate": trial_row["training.learning_rate"],
        }
        for window_name, weight in windows.items():
            values = []
            for seed in tuner.evaluation_seeds:
                rng = np.random.default_rng(seed + int(weight * 100))
                signal = weight + 0.02 * (params["hidden_size"] / 64) - params["learning_rate"] * 10
                noise = rng.normal(scale=0.001)
                values.append(signal + noise)
            expected_records.append(
                {
                    "trial": trial_row["trial"],
                    "window": window_name,
                    "mean": float(np.mean(values)),
                }
            )

    expected_df = pd.DataFrame(expected_records)
    ascending = False
    expected_df["rank"] = expected_df.groupby("window")["mean"].rank(method="min", ascending=ascending)
    expected_df = expected_df.sort_values(["window", "rank", "trial"]).reset_index(drop=True)

    ranking_subset = ranking_df[["trial", "window", "mean", "rank"]]

    pd.testing.assert_frame_equal(
        ranking_subset.sort_index(axis=1),
        expected_df.sort_index(axis=1),
        check_exact=False,
        atol=1e-6,
    )

    # Ensure the optimisation objective is consistent with the averaged window means
    computed_values = (
        expected_df.groupby("trial")["mean"].mean().sort_index().to_numpy()
    )
    np.testing.assert_allclose(results_df.sort_values("trial")["value"].to_numpy(), computed_values)


def test_tuner_respects_trial_limit(tmp_path: Path) -> None:
    base_model = ModelConfig(input_features=4, time_steps=16)
    base_trainer = TrainerConfig()
    windows = {"w0": 0.1, "w1": 0.2}
    search_space = SearchSpaceConfig(
        model={},
        training={},
        metric="val_score",
        direction="maximize",
        seeds=[0, 1, 2, 3, 4],
        top_k=1,
        sampler_seed=None,
    )

    tuner = HyperparameterTuner(
        _evaluation_fn,
        base_model_cfg=base_model,
        base_trainer_cfg=base_trainer,
        windows=windows,
        search_space=search_space,
        run_id="limit_case",
        output_dir=tmp_path / "artifacts",
        governance={"max_hpo_trials": {"temporal_transformer": 1}},
    )

    with pytest.raises(ValueError):
        tuner.optimize(n_trials=2)

