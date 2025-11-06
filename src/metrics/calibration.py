"""Calibration metrics for probabilistic forecasts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "CalibrationSummary",
    "crps_ensemble",
    "interval_coverage",
    "interval_coverage_error",
    "pit_values",
    "pit_histogram",
    "reliability_curve",
]


@dataclass(frozen=True)
class CalibrationSummary:
    """Container for calibration diagnostics."""

    nominal: np.ndarray
    observed: np.ndarray
    stderr: np.ndarray | None = None

    def as_dict(self) -> dict[str, np.ndarray | None]:
        return {"nominal": self.nominal, "observed": self.observed, "stderr": self.stderr}


def _to_2d_array(samples: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("Samples must form a 2D array of shape (n_observations, n_draws)")
    return array


def _to_1d_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    if array.ndim != 1:
        array = array.reshape(-1)
    return array


def crps_ensemble(y_true: Iterable[float] | np.ndarray, samples: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Return the CRPS for each observation given ensemble samples."""

    truth = _to_1d_array(y_true)
    ensemble = _to_2d_array(samples)
    if ensemble.shape[0] != truth.shape[0]:
        raise ValueError("`samples` and `y_true` must have the same number of rows")

    absolute_errors = np.abs(ensemble - truth[:, None])
    first_term = np.mean(absolute_errors, axis=1)

    pairwise_diffs = np.abs(ensemble[:, :, None] - ensemble[:, None, :])
    second_term = 0.5 * np.mean(pairwise_diffs, axis=(1, 2))
    return first_term - second_term


def interval_coverage(
    y_true: Iterable[float] | np.ndarray,
    lower: Iterable[float] | np.ndarray,
    upper: Iterable[float] | np.ndarray,
) -> float:
    """Compute empirical coverage of an interval forecast."""

    truth = _to_1d_array(y_true)
    lower_arr = _to_1d_array(lower)
    upper_arr = _to_1d_array(upper)
    if not (truth.size == lower_arr.size == upper_arr.size):
        raise ValueError("`y_true`, `lower`, and `upper` must be the same length")
    if np.any(lower_arr > upper_arr):
        raise ValueError("Lower bounds must not exceed upper bounds")

    covered = (truth >= lower_arr) & (truth <= upper_arr)
    return float(np.mean(covered))


def interval_coverage_error(
    y_true: Iterable[float] | np.ndarray,
    lower: Iterable[float] | np.ndarray,
    upper: Iterable[float] | np.ndarray,
    nominal: float,
) -> float:
    """Return the deviation between empirical and nominal coverage."""

    empirical = interval_coverage(y_true, lower, upper)
    return empirical - float(nominal)


def pit_values(
    y_true: Iterable[float] | np.ndarray,
    samples: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Compute PIT values from ensemble samples."""

    truth = _to_1d_array(y_true)
    ensemble = _to_2d_array(samples)
    if ensemble.shape[0] != truth.shape[0]:
        raise ValueError("`samples` and `y_true` must have the same number of rows")

    indicators = ensemble <= truth[:, None]
    return indicators.mean(axis=1)


def pit_histogram(
    y_true: Iterable[float] | np.ndarray,
    samples: np.ndarray | Sequence[Sequence[float]],
    *,
    bins: int | Sequence[float] = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PIT histogram counts and bin edges."""

    pit = pit_values(y_true, samples)
    counts, edges = np.histogram(pit, bins=bins, range=(0.0, 1.0))
    return counts, edges


def reliability_curve(
    y_true: Iterable[float] | np.ndarray,
    quantiles: dict[float, Iterable[float] | np.ndarray],
) -> CalibrationSummary:
    """Compute observed frequencies for forecast quantiles."""

    truth = _to_1d_array(y_true)
    if not quantiles:
        raise ValueError("`quantiles` must contain at least one entry")

    nominal_levels = np.array(sorted(quantiles.keys()), dtype=float)
    observed = []
    for level in nominal_levels:
        values = _to_1d_array(quantiles[level])
        if values.size != truth.size:
            raise ValueError("All quantile arrays must match the length of `y_true`")
        observed.append(np.mean(truth <= values))
    observed_arr = np.array(observed, dtype=float)
    n = float(truth.size)
    stderr = np.sqrt(np.maximum(nominal_levels * (1.0 - nominal_levels), 0.0) / n) if n > 0 else None
    return CalibrationSummary(nominal=nominal_levels, observed=observed_arr, stderr=stderr)
