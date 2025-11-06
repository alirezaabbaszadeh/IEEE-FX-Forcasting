from pathlib import Path

import numpy as np
import pandas as pd

from src.cli import _persist_split_audit
from src.splits.purged_cv import PurgedCVConfig, PurgedCVSplitter
from src.splits.walk_forward import WalkForwardConfig, WalkForwardSplitter


def test_walk_forward_splitter_diagnostics() -> None:
    index = pd.date_range("2022-01-01", periods=80, freq="1H")
    cfg = WalkForwardConfig(train_size=20, val_size=8, test_size=8, embargo=2, step=8)
    splitter = WalkForwardSplitter(cfg)

    splits = splitter.split(index)
    assert splits

    freq = index[1] - index[0]
    expected_gap = freq * (cfg.embargo + 1)

    for split in splits:
        diag = split.diagnostics
        if diag.embargo_gap_train_val is not None:
            assert diag.embargo_gap_train_val >= expected_gap
        if diag.embargo_gap_val_test is not None:
            assert diag.embargo_gap_val_test >= expected_gap
        records = split.records(pair="EURUSD", horizon="1H")
        assert {row["split"] for row in records} == {"train", "val", "test"}
        train_row = next(row for row in records if row["split"] == "train")
        assert train_row["size"] == cfg.train_size
        assert train_row["start"] is not None
        assert train_row["end"] is not None


def test_purged_cv_respects_embargo() -> None:
    index = pd.date_range("2022-03-01", periods=60, freq="30min")
    cfg = PurgedCVConfig(n_splits=3, test_size=6, embargo=2)
    splitter = PurgedCVSplitter(cfg)

    splits = splitter.split(index)
    assert len(splits) == 3

    freq = index[1] - index[0]
    expected_gap = freq * (cfg.embargo + 1)

    for split in splits:
        assert not np.intersect1d(split.train, split.test).size
        diag = split.diagnostics
        if diag.left_gap is not None:
            assert diag.left_gap >= expected_gap
        if diag.right_gap is not None:
            assert diag.right_gap >= expected_gap


def test_cli_writes_split_audit(tmp_path: Path) -> None:
    index = pd.date_range("2022-06-01", periods=40, freq="1H")
    cfg = WalkForwardConfig(train_size=12, val_size=6, test_size=6, embargo=1)
    splitter = WalkForwardSplitter(cfg)
    split = splitter.split(index)[0]

    dataset_metadata = {
        "pair": "EURUSD",
        "horizon": "1H",
        "window_id": 0,
        "split_records": split.records(pair="EURUSD", horizon="1H"),
    }

    _persist_split_audit(tmp_path, dataset_metadata)

    path = tmp_path / "splits.csv"
    assert path.exists()

    frame = pd.read_csv(path)
    assert set(frame["split"]) == {"train", "val", "test"}
    assert frame.loc[frame["split"] == "train", "size"].iloc[0] == cfg.train_size
    embargo_gaps = frame["embargo_gap_before"].dropna().unique()
    assert embargo_gaps.size >= 1
