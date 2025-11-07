from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.run import run_evaluation
from src.inference.stacking_purged import PurgedStackingConfig, PurgedStackingEnsembler


def _synthetic_predictions(num_rows: int = 6, *, include_test: bool = False) -> pd.DataFrame:
    base_time = pd.Timestamp("2024-01-01", tz="UTC")
    records: list[dict[str, object]] = []
    for idx in range(num_rows):
        ts = base_time + pd.Timedelta(minutes=30 * idx)
        split = "val"
        if include_test and idx >= num_rows // 2:
            split = "test"
        y_true = float(idx % 3)
        predictions = {
            "model_a": y_true,
            "model_b": y_true + (0.5 if split == "test" else 1.0),
        }
        for model, y_pred in predictions.items():
            records.append(
                {
                    "pair": "EURUSD",
                    "horizon": "1h",
                    "timestamp": ts,
                    "split": split,
                    "model": model,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
    return pd.DataFrame.from_records(records)


def test_purged_stacking_weights_prioritise_low_mse() -> None:
    frame = _synthetic_predictions(include_test=False)
    cfg = PurgedStackingConfig(
        base_models=("model_a", "model_b"),
        ensemble_name="stacked_fx",
        n_splits=3,
        embargo=1,
        validation_splits=("val",),
        apply_splits=("val",),
        min_validation=3,
    )
    blender = PurgedStackingEnsembler(cfg)
    result = blender.blend(frame)

    weights = result.weights
    assert not weights.empty
    sorted_weights = weights.sort_values("model").reset_index(drop=True)
    assert np.isclose(sorted_weights.loc[0, "weight"], 1.0)
    assert np.isclose(sorted_weights.loc[1, "weight"], 0.0)

    stacked = result.predictions
    assert not stacked.empty
    assert (stacked["model"] == "stacked_fx").all()
    assert np.allclose(stacked["y_pred"], stacked["y_true"])


def test_run_evaluation_appends_stacking_predictions(tmp_path: Path) -> None:
    frame = _synthetic_predictions(include_test=True)
    predictions_path = tmp_path / "predictions.csv"
    frame.to_csv(predictions_path, index=False)

    cfg = PurgedStackingConfig(
        base_models=("model_a", "model_b"),
        ensemble_name="stacked_fx",
        n_splits=3,
        embargo=1,
        validation_splits=("val",),
        apply_splits=("test",),
        min_validation=3,
    )

    metrics_path = run_evaluation(
        predictions_path=predictions_path,
        run_id="stacking_test",
        artifacts_dir=tmp_path,
        stacking_cfg=cfg,
    )

    metrics = pd.read_csv(metrics_path)
    assert (metrics["model"] == "stacked_fx").any()

    weights_path = tmp_path / "stacking_test" / "stacking_weights.csv"
    assert weights_path.exists()
    weights = pd.read_csv(weights_path)
    assert np.isclose(weights["weight"].sum(), 1.0)

    diagnostics_path = tmp_path / "stacking_test" / "stacking_fold_metrics.csv"
    assert diagnostics_path.exists()
    diagnostics = pd.read_csv(diagnostics_path)
    assert {"fold", "model", "mse"}.issubset(diagnostics.columns)
