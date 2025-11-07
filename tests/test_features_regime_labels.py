import numpy as np
import pandas as pd

from src.features import VolatilityRegimeConfig, compute_regime_features, label_volatility_regimes


def test_label_volatility_regimes_detects_stress_segment():
    calm = np.random.default_rng(0).normal(scale=0.05, size=120)
    stress = np.random.default_rng(1).normal(scale=0.5, size=60)
    series = pd.Series(np.concatenate([calm, stress]))
    config = VolatilityRegimeConfig(window=32, min_periods=8, stress_quantile=0.7)
    labels = label_volatility_regimes(series, config=config)
    assert len(labels) == len(series)
    stress_fraction = (labels.iloc[-60:] == "stress").mean()
    assert stress_fraction > 0.5


def test_compute_regime_features_returns_expected_columns():
    series = pd.Series(np.linspace(-0.1, 0.1, 50))
    features = compute_regime_features(series)
    expected = {"abs_return", "realised_volatility", "smoothed_volatility", "tail_risk", "stress_score"}
    assert expected.issubset(features.columns)
    assert len(features) == len(series)
