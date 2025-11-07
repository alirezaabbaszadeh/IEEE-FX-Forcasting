"""Purged cross-validated stacking ensemble with embargo-aware weights."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.splits import PurgedCVConfig, PurgedCVSplitter


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PurgedStackingConfig:
    """Configuration for generating purged stacking ensemble weights."""

    base_models: Sequence[str]
    ensemble_name: str = "stacking_purged"
    n_splits: int = 5
    embargo: int = 0
    test_size: int | None = None
    validation_splits: Sequence[str] = ("val",)
    apply_splits: Sequence[str] = ("val", "test")
    weight_floor: float = 0.0
    min_validation: int = 1
    timestamp_column: str = "timestamp"
    model_column: str = "model"
    target_column: str = "y_true"
    prediction_column: str = "y_pred"
    split_column: str = "split"
    pair_column: str = "pair"
    horizon_column: str = "horizon"

    def __post_init__(self) -> None:
        models = tuple(str(model) for model in self.base_models if str(model))
        if not models:
            raise ValueError("base_models must contain at least one model identifier")
        object.__setattr__(self, "base_models", models)
        if int(self.n_splits) <= 1:
            raise ValueError("n_splits must be greater than 1 for purged stacking")
        if int(self.embargo) < 0:
            raise ValueError("embargo cannot be negative")
        if self.test_size is not None and int(self.test_size) <= 0:
            raise ValueError("test_size must be a positive integer when provided")
        if float(self.weight_floor) < 0.0:
            raise ValueError("weight_floor cannot be negative")
        if int(self.min_validation) <= 0:
            raise ValueError("min_validation must be a positive integer")
        if not self.validation_splits:
            raise ValueError("validation_splits must contain at least one split label")

    @property
    def validation_labels(self) -> set[str]:
        return {str(value).lower() for value in self.validation_splits}

    @property
    def apply_labels(self) -> set[str]:
        return {str(value).lower() for value in self.apply_splits}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "PurgedStackingConfig":
        data = dict(payload)
        if "base_models" in data:
            data["base_models"] = tuple(str(item) for item in data["base_models"])
        if "validation_splits" in data:
            data["validation_splits"] = tuple(str(item) for item in data["validation_splits"])
        if "apply_splits" in data:
            data["apply_splits"] = tuple(str(item) for item in data["apply_splits"])
        return cls(**data)


@dataclass(frozen=True)
class PurgedStackingResult:
    """Container for ensemble predictions and diagnostic weight tables."""

    predictions: pd.DataFrame
    weights: pd.DataFrame
    fold_diagnostics: pd.DataFrame


class PurgedStackingEnsembler:
    """Generate leakage-aware stacking predictions by weighting base models."""

    REQUIRED_COLUMNS = {"pair", "horizon", "timestamp", "model", "y_true", "y_pred"}

    def __init__(self, config: PurgedStackingConfig):
        self.cfg = config

    def blend(self, predictions: pd.DataFrame) -> PurgedStackingResult:
        """Return stacked predictions and diagnostics derived from validation folds."""

        frame = self._prepare_frame(predictions)
        ensemble_rows: list[pd.DataFrame] = []
        weight_rows: list[dict[str, object]] = []
        diagnostics_rows: list[dict[str, object]] = []

        for (pair, horizon), group in frame.groupby(
            [self.cfg.pair_column, self.cfg.horizon_column], sort=False
        ):
            blended, weights, diagnostics = self._blend_group(pair, horizon, group)
            if blended is not None and not blended.empty:
                ensemble_rows.append(blended)
            weight_rows.extend(weights)
            diagnostics_rows.extend(diagnostics)

        ensemble_df = (
            pd.concat(ensemble_rows, ignore_index=True)
            if ensemble_rows
            else pd.DataFrame(columns=list(predictions.columns))
        )
        weights_df = pd.DataFrame.from_records(weight_rows)
        diagnostics_df = pd.DataFrame.from_records(diagnostics_rows)
        return PurgedStackingResult(ensemble_df, weights_df, diagnostics_df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_frame(self, predictions: pd.DataFrame) -> pd.DataFrame:
        missing = self.REQUIRED_COLUMNS.difference(predictions.columns)
        if missing:
            raise KeyError(f"Missing required columns for stacking: {sorted(missing)}")
        frame = predictions.copy()
        frame[self.cfg.model_column] = frame[self.cfg.model_column].astype(str)
        return frame

    def _blend_group(
        self, pair: str, horizon: object, group: pd.DataFrame
    ) -> tuple[pd.DataFrame | None, list[dict[str, object]], list[dict[str, object]]]:
        subset = group[group[self.cfg.model_column].isin(self.cfg.base_models)]
        if subset.empty:
            return None, [], []

        index_columns = [
            column
            for column in subset.columns
            if column not in {self.cfg.model_column, self.cfg.prediction_column}
        ]
        wide = (
            subset.pivot_table(
                index=index_columns,
                columns=self.cfg.model_column,
                values=self.cfg.prediction_column,
            )
            .reset_index()
            .sort_values(self.cfg.timestamp_column)
        )

        for model in self.cfg.base_models:
            if model not in wide.columns:
                raise KeyError(
                    f"Model '{model}' missing for pair={pair!r}, horizon={horizon!r}"
                )

        if self.cfg.split_column in wide.columns:
            validation_mask = (
                wide[self.cfg.split_column].astype(str).str.lower().isin(self.cfg.validation_labels)
            )
        else:
            validation_mask = np.ones(len(wide), dtype=bool)

        validation = wide.loc[validation_mask].copy()
        if validation.empty or len(validation) < int(self.cfg.min_validation):
            return None, [], []

        timestamps = pd.to_datetime(
            validation[self.cfg.timestamp_column], utc=True, errors="coerce"
        )
        if timestamps.isnull().any():
            raise ValueError(
                f"Non-parsable timestamps detected for pair={pair!r}, horizon={horizon!r}"
            )

        features = validation.loc[:, list(self.cfg.base_models)].to_numpy(dtype=float)
        targets = validation[self.cfg.target_column].to_numpy(dtype=float)

        splitter = PurgedCVSplitter(
            PurgedCVConfig(
                n_splits=int(self.cfg.n_splits),
                test_size=self.cfg.test_size,
                embargo=int(self.cfg.embargo),
            )
        )
        try:
            splits = splitter.split(list(timestamps))
        except ValueError as exc:
            LOGGER.warning(
                "Falling back to single-fold stacking for pair=%s horizon=%s: %s",
                pair,
                horizon,
                exc,
            )
            splits = [
                SimpleNamespace(
                    fold=0,
                    test=np.arange(len(validation), dtype=int),
                )
            ]

        mse_sums = np.zeros(len(self.cfg.base_models), dtype=float)
        sample_counts = np.zeros(len(self.cfg.base_models), dtype=float)
        diagnostics_rows: list[dict[str, object]] = []

        for split in splits:
            test_idx = split.test
            if test_idx.size == 0:
                continue
            fold_preds = features[test_idx]
            fold_targets = targets[test_idx]
            residuals = fold_preds - fold_targets[:, None]
            squared = residuals**2
            mse_fold = squared.mean(axis=0)
            mse_sums += squared.sum(axis=0)
            sample_counts += squared.shape[0]
            for model_idx, model_name in enumerate(self.cfg.base_models):
                diagnostics_rows.append(
                    {
                        self.cfg.pair_column: pair,
                        self.cfg.horizon_column: horizon,
                        "fold": int(split.fold),
                        "model": model_name,
                        "mse": float(mse_fold[model_idx]),
                        "samples": int(squared.shape[0]),
                    }
                )

        valid_counts = sample_counts > 0
        if not np.any(valid_counts):
            return None, [], diagnostics_rows

        mse = np.divide(
            mse_sums,
            sample_counts,
            out=np.full_like(mse_sums, np.inf, dtype=float),
            where=sample_counts > 0,
        )

        weights = self._weights_from_mse(mse)

        weight_rows = [
            {
                self.cfg.pair_column: pair,
                self.cfg.horizon_column: horizon,
                "model": model_name,
                "weight": float(weights[idx]),
                "mse": float(mse[idx]),
                "validation_samples": int(sample_counts[idx]),
            }
            for idx, model_name in enumerate(self.cfg.base_models)
        ]

        if self.cfg.split_column in wide.columns:
            apply_mask = (
                wide[self.cfg.split_column].astype(str).str.lower().isin(self.cfg.apply_labels)
            )
        else:
            apply_mask = np.ones(len(wide), dtype=bool)

        targets_full = wide.loc[apply_mask, self.cfg.target_column].to_numpy(dtype=float)
        predictors_full = wide.loc[apply_mask, list(self.cfg.base_models)].to_numpy(dtype=float)
        blended_preds = predictors_full @ weights

        blended = wide.loc[apply_mask, index_columns].copy()
        blended[self.cfg.prediction_column] = blended_preds
        blended[self.cfg.model_column] = self.cfg.ensemble_name
        blended[self.cfg.target_column] = targets_full

        return blended, weight_rows, diagnostics_rows

    def _weights_from_mse(self, mse: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            finite_mask = np.isfinite(mse)
            if not np.any(finite_mask):
                return np.full(len(mse), 1.0 / len(mse), dtype=float)

            zero_mask = mse == 0.0
            if np.any(zero_mask):
                weights = np.zeros_like(mse, dtype=float)
                weights[zero_mask] = 1.0 / zero_mask.sum()
                return weights

            inverse = np.divide(1.0, mse, out=np.zeros_like(mse), where=finite_mask)

        if float(self.cfg.weight_floor) > 0.0:
            inverse = np.maximum(inverse, float(self.cfg.weight_floor))

        total = inverse.sum()
        if total <= 0.0:
            return np.full(len(mse), 1.0 / len(mse), dtype=float)
        return inverse / total


__all__ = [
    "PurgedStackingConfig",
    "PurgedStackingEnsembler",
    "PurgedStackingResult",
]

