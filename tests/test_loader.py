import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.loader import (
    ChronologicalDataLoader,
    FEATURE_COLUMNS,
    LegacyLoaderOptions,
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


def test_loader_supports_custom_feature_and_target_columns(tmp_path):
    rows = 10
    base = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "Bid": base + 1,
            "Ask": base + 2,
            "Spread": base * 0.1,
            "Target": base + 10,
        }
    )
    csv_path = _write_csv(tmp_path, frame)

    loader = ChronologicalDataLoader(
        str(csv_path),
        time_steps=2,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        feature_columns=("Bid", "Spread"),
        target_column="Target",
    )
    artifacts = loader.load()

    assert artifacts.feature_columns == ("Bid", "Spread")
    assert artifacts.target_column == "Target"
    assert artifacts.train.features.shape[2] == 2

    train_slice = frame.iloc[: int(0.6 * rows)]
    expected_feature_mean = train_slice[["Bid", "Spread"]].mean().values
    expected_target_mean = train_slice["Target"].mean()

    np.testing.assert_allclose(artifacts.feature_scaler.mean_, expected_feature_mean)
    np.testing.assert_allclose(artifacts.target_scaler.mean_, np.array([expected_target_mean]))


def test_loader_handles_timestamp_deduplication(tmp_path):
    timestamps = [
        "2024-03-10 00:00",
        "2024-03-10 01:00",
        "2024-03-10 01:00",
        "2024-03-10 02:00",
        "2024-03-10 03:00",
        "2024-03-10 03:00",
        "2024-03-10 04:00",
        "2024-03-10 05:00",
    ]
    values = np.arange(len(timestamps), dtype=float)
    frame = pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Open": values,
            "High": values + 1,
            "Low": values + 2,
            "Close": values + 3,
        }
    )
    csv_path = _write_csv(tmp_path, frame)

    options = LegacyLoaderOptions(
        timestamp_column="Timestamp",
        sort_by_timestamp=True,
        drop_duplicate_timestamps=True,
        duplicate_keep="last",
    )
    loader = ChronologicalDataLoader(
        str(csv_path),
        time_steps=2,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        options=options,
    )
    artifacts = loader.load()

    expected_frame = frame.copy()
    expected_frame["Timestamp"] = pd.to_datetime(expected_frame["Timestamp"])
    expected_frame = expected_frame.dropna()
    expected_frame = expected_frame.sort_values("Timestamp", kind="mergesort")
    expected_frame = expected_frame.drop_duplicates(
        subset=["Timestamp"], keep="last"
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(loader._load_frame(), expected_frame)

    train_end = int(0.5 * len(expected_frame))
    val_end = int((0.5 + 0.25) * len(expected_frame))
    assert artifacts.split_metadata.train == SplitBounds(0, train_end)
    assert artifacts.split_metadata.val == SplitBounds(train_end, val_end)
    assert artifacts.split_metadata.test == SplitBounds(val_end, len(expected_frame))

    assert len(expected_frame) == len(frame) - 2  # two duplicates removed
    expected_train_sequences = max(0, train_end - loader.time_steps)
    assert artifacts.train.features.shape[0] == expected_train_sequences

    # The duplicate hour at 01:00 should keep the last occurrence (Close value 5)
    deduped_close_values = expected_frame.loc[:, "Close"].to_numpy()
    assert 5.0 in deduped_close_values
    assert 4.0 not in deduped_close_values


def test_get_data_returns_legacy_tuple(tmp_path):
    rows = 8
    base = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "Open": base,
            "High": base + 1,
            "Low": base + 2,
            "Close": base + 3,
        }
    )
    csv_path = _write_csv(tmp_path, frame)

    loader = ChronologicalDataLoader(
        str(csv_path), time_steps=2, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25
    )
    tuple_data = loader.get_data()

    assert len(tuple_data) == 7
    artifacts = loader.load()

    for tuple_array, artifact_array in zip(
        tuple_data[:6],
        [
            artifacts.train.features,
            artifacts.val.features,
            artifacts.test.features,
            artifacts.train.target,
            artifacts.val.target,
            artifacts.test.target,
        ],
    ):
        np.testing.assert_allclose(tuple_array, artifact_array)

    assert isinstance(tuple_data[6], StandardScaler)


def test_legacy_loader_options_rejects_invalid_override():
    options = LegacyLoaderOptions()
    with pytest.raises(TypeError, match="Invalid loader option"):
        options.with_overrides(nonexistent=True)
