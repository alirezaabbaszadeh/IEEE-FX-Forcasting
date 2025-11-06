from __future__ import annotations

import numpy as np
import pytest

from src.metrics.calibration import (
    CalibrationSummary,
    crps_ensemble,
    interval_coverage,
    interval_coverage_error,
    pit_values,
    reliability_curve,
)


def test_crps_ensemble_matches_manual_calculation() -> None:
    y_true = np.array([0.0, 1.0])
    samples = np.array([[0.0, 1.0], [0.0, 1.0]])

    expected = np.array([0.25, 0.25])
    result = crps_ensemble(y_true, samples)

    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_interval_coverage_error_is_zero_when_calibrated() -> None:
    y_true = np.array([0.0, 0.5, 1.0, 1.5])
    lower = np.array([-0.5, 0.0, 0.5, 1.0])
    upper = np.array([0.5, 1.0, 1.5, 2.0])

    coverage = interval_coverage(y_true, lower, upper)
    error = interval_coverage_error(y_true, lower, upper, nominal=0.75)

    assert coverage == pytest.approx(0.75)
    assert error == pytest.approx(0.0)


def test_reliability_curve_orders_nominal_levels() -> None:
    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    quantiles = {
        0.9: np.array([0.1, 1.1, 2.1, 3.1]),
        0.5: np.array([-0.2, 0.8, 1.8, 2.8]),
        0.1: np.array([-0.5, 0.5, 1.5, 2.5]),
    }

    summary = reliability_curve(y_true, quantiles)
    assert isinstance(summary, CalibrationSummary)
    assert np.all(np.diff(summary.nominal) > 0)
    assert summary.observed.shape == summary.nominal.shape


def test_pit_values_recover_uniform_distribution() -> None:
    rng = np.random.default_rng(7)
    y_true = rng.normal(size=256)
    samples = rng.normal(loc=y_true[:, None], scale=0.1, size=(y_true.size, 100))

    pit = pit_values(y_true, samples)
    assert pit.shape[0] == y_true.size
    # Highly concentrated predictions should yield PIT concentrated near 0.5
    assert np.mean(pit) == pytest.approx(0.5, abs=0.05)
