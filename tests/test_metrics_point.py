from __future__ import annotations

import numpy as np
import pytest

from src.metrics.point import mase


def test_mase_uses_in_sample_naive_forecast():
    y_true = np.array([10.0, 12.0, 11.0, 15.0, 14.0])
    y_pred = np.array([9.0, 11.5, 12.0, 14.0, 13.5])

    naive_errors = np.abs(np.diff(y_true))
    expected_scale = naive_errors.mean()
    numerator = np.mean(np.abs(y_pred - y_true))

    result = mase(y_true, y_pred)

    assert result == pytest.approx(numerator / expected_scale)
