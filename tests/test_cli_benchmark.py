from __future__ import annotations

import sys

import pytest

import src.cli


def _reset_benchmark_mode() -> None:
    src.cli._set_benchmark_mode(None)


def test_main_enables_smoke_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, int] = {}

    def _fake_single() -> None:
        calls["single"] = calls.get("single", 0) + 1

    def _fake_multi() -> None:
        calls["multi"] = calls.get("multi", 0) + 1

    monkeypatch.setattr(src.cli, "_hydra_single", _fake_single)
    monkeypatch.setattr(src.cli, "_hydra_multirun", _fake_multi)
    monkeypatch.setattr(sys, "argv", ["prog", "--benchmark-smoke"])

    try:
        src.cli.main()
        assert src.cli._get_benchmark_mode() == "smoke"
    finally:
        _reset_benchmark_mode()

    assert calls.get("single") == 1
    assert "multi" not in calls


def test_main_enables_full_benchmark_for_multirun(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, int] = {}

    def _fake_single() -> None:
        calls["single"] = calls.get("single", 0) + 1

    def _fake_multi() -> None:
        calls["multi"] = calls.get("multi", 0) + 1

    monkeypatch.setattr(src.cli, "_hydra_single", _fake_single)
    monkeypatch.setattr(src.cli, "_hydra_multirun", _fake_multi)
    monkeypatch.setattr(sys, "argv", ["prog", "--multirun", "--benchmark-full"])

    try:
        src.cli.main()
        assert src.cli._get_benchmark_mode() == "full"
    finally:
        _reset_benchmark_mode()

    assert calls.get("multi") == 1
    assert "single" not in calls


def test_main_rejects_conflicting_benchmark_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--benchmark-smoke", "--benchmark-full"])

    with pytest.raises(SystemExit) as exc:
        src.cli.main()

    assert "cannot be used together" in str(exc.value)
    assert src.cli._get_benchmark_mode() is None
