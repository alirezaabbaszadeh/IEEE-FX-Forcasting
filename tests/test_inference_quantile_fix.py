from __future__ import annotations

import numpy as np
import pandas as pd

from src.inference.quantile_fix import fix_quantile_frame, project_monotonic_quantiles


def test_project_monotonic_quantiles_synthetic_crossings() -> None:
    rng = np.random.default_rng(42)
    base = np.sort(rng.normal(size=(32, 4)), axis=1)
    perturbed = base.copy()
    perturbed[:, 2] -= 0.8
    perturbed[:, 3] -= 1.5

    corrected = project_monotonic_quantiles(perturbed)
    assert isinstance(corrected, np.ndarray)
    diffs = np.diff(corrected, axis=1)
    assert np.all(diffs >= -1e-12)

    original_error = np.linalg.norm(perturbed - base)
    corrected_error = np.linalg.norm(corrected - base)
    assert corrected_error <= original_error + 1e-9


def test_fix_quantile_frame_corrects_crossings() -> None:
    frame = pd.DataFrame(
        {
            "quantile_0.10": [0.1, -0.2, 0.0],
            "quantile_0.50": [0.0, 0.0, 0.1],
            "quantile_0.90": [-0.1, -0.4, 0.05],
        }
    )

    fixed = fix_quantile_frame(frame, ["quantile_0.10", "quantile_0.50", "quantile_0.90"])
    assert np.all(np.diff(fixed.to_numpy(dtype=float), axis=1) >= -1e-12)
    np.testing.assert_allclose(fixed.iloc[0].to_numpy(dtype=float), [0.0, 0.0, 0.0])
