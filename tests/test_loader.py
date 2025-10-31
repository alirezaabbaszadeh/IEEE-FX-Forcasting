import numpy as np
import pandas as pd
import pytest

from src.data.loader import (
    ChronologicalDataLoader,
    FEATURE_COLUMNS,
    SplitBounds,
)


def _write_csv(tmp_path, frame):
    path = tmp_path / "series.csv"
    frame.to_csv(path, index=False)
    return path


def test_loader_creates_sequences_and_metadata(tmp_path):
    rows = 12
    base = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "Open": base + 1,
            "High": base + 2,
            "Low": base + 3,
            "Close": base + 4,
        }
    )
    csv_path = _write_csv(tmp_path, frame)

    loader = ChronologicalDataLoader(
        str(csv_path), time_steps=2, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25
    )
    artifacts = loader.load()

    assert artifacts.train.features.shape == (4, 2, len(FEATURE_COLUMNS))
    assert artifacts.train.target.shape == (4, 1)
    assert artifacts.val.features.shape == (1, 2, len(FEATURE_COLUMNS))
    assert artifacts.test.features.shape == (1, 2, len(FEATURE_COLUMNS))

    assert artifacts.split_metadata.train == SplitBounds(0, 6)
    assert artifacts.split_metadata.val == SplitBounds(6, 9)
    assert artifacts.split_metadata.test == SplitBounds(9, 12)

    train_slice = frame.iloc[:6]
    expected_feature_mean = train_slice[list(FEATURE_COLUMNS)].mean().values
    expected_target_mean = train_slice["Close"].mean()

    np.testing.assert_allclose(artifacts.feature_scaler.mean_, expected_feature_mean)
    np.testing.assert_allclose(artifacts.target_scaler.mean_, np.array([expected_target_mean]))


def test_loader_handles_sequences_shorter_than_time_steps(tmp_path):
    rows = 6
    base = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "Open": base,
            "High": base,
            "Low": base,
            "Close": base,
        }
    )
    csv_path = _write_csv(tmp_path, frame)

    loader = ChronologicalDataLoader(
        str(csv_path), time_steps=5, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
    )
    artifacts = loader.load()

    assert artifacts.train.features.shape == (0, 5, len(FEATURE_COLUMNS))
    assert artifacts.train.target.shape == (0, 1)
    assert artifacts.val.features.shape == (0, 5, len(FEATURE_COLUMNS))
    assert artifacts.test.features.shape == (0, 5, len(FEATURE_COLUMNS))


def test_scalers_fit_only_on_training_data(tmp_path):
    train_values = np.array([0.0, 1.0, 2.0, 3.0])
    val_values = np.array([100.0, 101.0])
    test_values = np.array([200.0, 201.0])
    values = np.concatenate([train_values, val_values, test_values])

    frame = pd.DataFrame(
        {
            "Open": values + 10,
            "High": values + 20,
            "Low": values + 30,
            "Close": values + 40,
        }
    )
    csv_path = _write_csv(tmp_path, frame)

    loader = ChronologicalDataLoader(
        str(csv_path), time_steps=1, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25
    )
    artifacts = loader.load()

    train_mean = train_values.mean()
    expected_feature_mean = np.array([
        train_mean + 10,
        train_mean + 20,
        train_mean + 30,
        train_mean + 40,
    ])
    expected_target_mean = np.array([train_mean + 40])
    train_var = np.var(train_values, ddof=0)
    expected_feature_var = np.full(len(FEATURE_COLUMNS), train_var)
    expected_target_var = np.array([train_var])

    np.testing.assert_allclose(artifacts.feature_scaler.mean_, expected_feature_mean)
    np.testing.assert_allclose(artifacts.feature_scaler.var_, expected_feature_var)
    np.testing.assert_allclose(artifacts.target_scaler.mean_, expected_target_mean)
    np.testing.assert_allclose(artifacts.target_scaler.var_, expected_target_var)

    std = np.sqrt(train_var)
    expected_val_open = (110 - expected_feature_mean[0]) / std
    assert artifacts.val.features.shape == (1, 1, len(FEATURE_COLUMNS))
    np.testing.assert_allclose(artifacts.val.features[0, 0, 0], expected_val_open)


def test_raises_when_training_split_empty(tmp_path):
    frame = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.0, 2.0],
            "Low": [1.0, 2.0],
            "Close": [1.0, 2.0],
        }
    )
    csv_path = _write_csv(tmp_path, frame)

    loader = ChronologicalDataLoader(
        str(csv_path), time_steps=2, train_ratio=0.0, val_ratio=0.5, test_ratio=0.5
    )

    with pytest.raises(ValueError, match="Training dataset is empty"):
        loader.load()
