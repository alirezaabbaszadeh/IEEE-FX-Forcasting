from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis.stats import (
    bootstrap_confidence_interval,
    construct_dm_comparisons,
    dunn_posthoc,
    hansen_model_confidence_set,
    tukey_hsd,
)

try:  # pragma: no cover - optional reference dependency
    import scipy.stats as scipy_reference  # type: ignore

    HAS_SCIPY = True
except ModuleNotFoundError:  # pragma: no cover - SciPy not installed
    scipy_reference = None
    HAS_SCIPY = False


def test_bootstrap_ci_reproducibility() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    res_a = bootstrap_confidence_interval(values, n_resamples=500, random_state=123)
    res_b = bootstrap_confidence_interval(values, n_resamples=500, random_state=123)
    assert res_a == res_b


def _holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda idx: p_values[idx])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        running_max = max(running_max, p_values[idx] * factor)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def test_dunn_posthoc_holm_adjustment() -> None:
    metrics = {
        "model_a": [1.0, 2.0, 3.0, 2.5],
        "model_b": [2.5, 2.7, 2.9, 3.1],
        "model_c": [0.5, 0.6, 0.4, 0.7],
    }
    results = dunn_posthoc(metrics, alpha=0.05)
    raw = [row["p_value"] for row in results]
    expected = _holm_adjust(raw)
    adjusted = [row["p_adjusted"] for row in results]
    assert pytest.approx(expected) == adjusted


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy reference unavailable")
def test_tukey_hsd_matches_scipy() -> None:
    metrics = {
        "baseline": np.array([0.55, 0.56, 0.54, 0.57]),
        "model_b": np.array([0.60, 0.59, 0.61, 0.58]),
        "model_c": np.array([0.50, 0.51, 0.49, 0.52]),
    }

    ours = tukey_hsd(metrics, alpha=0.05)
    reference = scipy_reference.tukey_hsd(*metrics.values())
    names = list(metrics.keys())
    ref_map: dict[tuple[str, str], float] = {}
    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if j <= i:
                continue
            ref_map[(name_a, name_b)] = float(reference.pvalue[i, j])
    for row in ours:
        key = (row["model_a"], row["model_b"])
        assert math.isclose(row["p_value"], ref_map[key], rel_tol=1e-5)


def test_model_confidence_set_includes_top_model() -> None:
    metrics = {
        "model_a": np.array([0.5, 0.55, 0.6, 0.58]),
        "model_b": np.array([0.52, 0.53, 0.51, 0.50]),
        "model_c": np.array([0.45, 0.47, 0.46, 0.44]),
    }
    results = hansen_model_confidence_set(metrics, n_bootstrap=200, random_state=42)
    best_model = max(metrics, key=lambda name: np.mean(metrics[name]))
    included = {row["model"]: row["included"] for row in results}
    assert included[best_model]


def test_construct_dm_comparisons_generates_pairs() -> None:
    repo = {
        "pair": ["A", "A", "A", "A"],
        "horizon": ["1d", "1d", "1d", "1d"],
        "model": ["m1", "m1", "m2", "m2"],
        "seed": ["0", "1", "0", "1"],
        "value": [0.1, 0.2, 0.3, 0.4],
    }
    df = construct_dm_comparisons(pd.DataFrame(repo))
    assert {"model_a", "model_b", "dm_stat", "p_value"}.issubset(df.columns)
    assert len(df) == 1
