"""Utilities for enforcing monotonic quantile estimates during inference."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - keep lightweight when torch missing
    torch = None  # type: ignore[assignment]


def _pava_1d(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Return the isotonic projection of a 1D array via pool-adjacent violators."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("_pava_1d expects a 1D array")
    if array.size == 0:
        return array.copy()

    if weights is None:
        weight_array = np.ones_like(array, dtype=float)
    else:
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.shape != array.shape:
            raise ValueError("weights must match the shape of values")
        if np.any(weight_array < 0):
            raise ValueError("weights cannot contain negative entries")

    blocks: list[tuple[float, float, int, int]] = []
    for idx, value in enumerate(array):
        block_value = float(value)
        block_weight = float(weight_array[idx])
        start = idx
        end = idx
        while blocks and blocks[-1][0] > block_value:
            prev_value, prev_weight, prev_start, _ = blocks.pop()
            total_weight = prev_weight + block_weight
            if total_weight == 0:
                block_value = 0.0
            else:
                block_value = (prev_value * prev_weight + block_value * block_weight) / total_weight
            block_weight = total_weight
            start = prev_start
        blocks.append((block_value, block_weight, start, end))

    projection = np.empty_like(array, dtype=float)
    for value, _, start, end in blocks:
        projection[start : end + 1] = value
    return projection


def _prepare_weights(
    weights: np.ndarray | Sequence[float] | None, rows: int, columns: int
) -> np.ndarray | None:
    if weights is None:
        return None
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.ndim == 1:
        if weight_array.shape[0] != columns:
            raise ValueError("1D weights must align with the quantile axis length")
        return np.broadcast_to(weight_array, (rows, columns))
    if weight_array.shape != (rows, columns):
        raise ValueError("weights must either be 1D or match the target shape")
    return weight_array


def _project_numpy(
    array: np.ndarray,
    *,
    axis: int,
    weights: np.ndarray | Sequence[float] | None = None,
) -> np.ndarray:
    moved = np.moveaxis(array, axis, -1)
    if moved.size == 0:
        return array.copy()
    flat = moved.reshape(-1, moved.shape[-1])
    weight_matrix = _prepare_weights(weights, flat.shape[0], flat.shape[1])
    result = np.empty_like(flat, dtype=float)
    for idx, row in enumerate(flat):
        row_weights = None if weight_matrix is None else weight_matrix[idx]
        result[idx] = _pava_1d(row, row_weights)
    reshaped = result.reshape(moved.shape)
    return np.moveaxis(reshaped, -1, axis)


def project_monotonic_quantiles(
    values: np.ndarray | "torch.Tensor",
    *,
    axis: int = -1,
    weights: np.ndarray | Sequence[float] | None = None,
) -> np.ndarray | "torch.Tensor":
    """Project quantile estimates onto the monotonic cone via isotonic regression."""

    if torch is not None and isinstance(values, torch.Tensor):
        if values.numel() == 0:
            return values.clone()
        with torch.no_grad():
            numpy_values = values.detach().cpu().numpy()
            projected = _project_numpy(numpy_values, axis=axis, weights=weights)
        return values.new_tensor(projected)

    array = np.asarray(values, dtype=float)
    projected_array = _project_numpy(array, axis=axis, weights=weights)
    return projected_array


def fix_quantile_frame(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Return a frame where `columns` are monotonic across each row."""

    column_list = [col for col in columns]
    if not column_list:
        return frame if inplace else frame.copy()
    missing = [col for col in column_list if col not in frame.columns]
    if missing:
        raise KeyError(f"Columns not present in frame: {sorted(missing)}")

    target = frame if inplace else frame.copy()
    if target.empty:
        return target

    data = target.loc[:, column_list].to_numpy(dtype=float)
    corrected = project_monotonic_quantiles(data, axis=-1)
    assert isinstance(corrected, np.ndarray)
    for idx, column in enumerate(column_list):
        target[column] = corrected[:, idx]
    return target


def resolve_quantile_columns(columns: Iterable[str]) -> list[str]:
    """Identify quantile-like columns within an iterable of column names."""

    quantile_columns: list[tuple[float, str]] = []
    for column in columns:
        if not column.startswith("quantile_"):
            continue
        level_text = column.split("quantile_")[1]
        try:
            level = float(level_text)
        except ValueError:
            continue
        quantile_columns.append((level, column))
    quantile_columns.sort(key=lambda item: item[0])
    return [column for _, column in quantile_columns]

