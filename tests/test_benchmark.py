from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # type: ignore # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # type: ignore # noqa: E402

from scripts.benchmark import (
    BenchmarkResult,
    BenchmarkThresholds,
    InferenceMetrics,
    TrainingMetrics,
    capture_environment_lockfiles,
    enforce_thresholds,
    run_benchmark,
)
from src.analysis.benchmark import benchmark_model, save_report


def test_capture_environment_lockfiles_creates_expected_files(tmp_path: Path) -> None:
    info = capture_environment_lockfiles(tmp_path)
    names = {entry["name"] for entry in info["lockfiles"]}
    assert names == {"conda", "pip", "cuda"}
    for record in info["lockfiles"]:
        file_path = Path(record["path"])
        assert file_path.exists()
        assert len(record["sha256"]) == 64


def test_enforce_thresholds_detects_regressions() -> None:
    training = TrainingMetrics(
        elapsed_seconds=1.0,
        epochs=1,
        samples_processed=100,
        throughput_samples_per_sec=100.0,
        cpu_rss_mb=200.0,
        cpu_maxrss_mb=250.0,
        gpu_peak_mb=None,
    )
    inference = InferenceMetrics(
        batch_size=32,
        latency_ms=[10.0, 11.0, 9.5],
        mean_latency_ms=10.166666,
        median_latency_ms=10.0,
        p95_latency_ms=11.0,
        cpu_rss_mb=210.0,
        gpu_peak_mb=None,
    )
    result = BenchmarkResult(
        training=training,
        inference=inference,
        metadata={},
        environment={},
        config_path="configs/default.yaml",
        config_hash="abc123",
        verification_hash="deadbeef",
    )

    thresholds = BenchmarkThresholds(
        min_train_throughput=150.0,
        max_inference_latency_ms=9.0,
    )
    with pytest.raises(RuntimeError) as exc:
        enforce_thresholds(result, thresholds)
    message = str(exc.value)
    assert "Training throughput" in message
    assert "Inference latency" in message


def test_enforce_thresholds_passes_when_within_limits() -> None:
    training = TrainingMetrics(
        elapsed_seconds=1.0,
        epochs=1,
        samples_processed=200,
        throughput_samples_per_sec=200.0,
        cpu_rss_mb=200.0,
        cpu_maxrss_mb=250.0,
        gpu_peak_mb=100.0,
    )
    inference = InferenceMetrics(
        batch_size=16,
        latency_ms=[5.0, 5.5, 4.8],
        mean_latency_ms=5.1,
        median_latency_ms=5.0,
        p95_latency_ms=5.5,
        cpu_rss_mb=205.0,
        gpu_peak_mb=120.0,
    )
    result = BenchmarkResult(
        training=training,
        inference=inference,
        metadata={},
        environment={},
        config_path="configs/default.yaml",
        config_hash="abc123",
        verification_hash="deadbeef",
    )

    thresholds = BenchmarkThresholds(
        min_train_throughput=150.0,
        max_inference_latency_ms=10.0,
        max_train_gpu_mb=150.0,
        max_inference_gpu_mb=200.0,
    )
    enforce_thresholds(result, thresholds)


def test_run_benchmark_produces_verifiable_metrics(tmp_path: Path) -> None:
    result = run_benchmark(
        Path("configs/default.yaml"),
        tmp_path,
        train_warmup_epochs=0,
        inference_warmup_iters=0,
        inference_measure_iters=1,
    )

    assert len(result.verification_hash) == 64
    enforce_thresholds(
        result,
        BenchmarkThresholds(
            min_train_throughput=10.0,
            max_inference_latency_ms=200.0,
        ),
    )


def test_benchmark_model_records_latency_percentiles(tmp_path: Path) -> None:
    torch.manual_seed(0)
    dataset = TensorDataset(torch.randn(24, 4), torch.randn(24, 1))
    dataloader = DataLoader(dataset, batch_size=8)
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))

    report = benchmark_model(
        model,
        dataloader,
        warmup_steps=1,
        measure_steps=3,
        dataloader_label="val",
    )

    metrics = report.metrics
    assert metrics.latency_ms_samples
    assert len(metrics.latency_ms_samples) == 3
    assert metrics.latency_p95_ms >= metrics.latency_p50_ms
    assert metrics.latency_max_ms >= metrics.latency_min_ms
    assert metrics.warmup_time_s >= 0
    assert metrics.throughput_samples_per_sec > 0
    assert report.settings.dataloader == "val"

    paths = save_report(report, tmp_path / "benchmarks", stem="inference")
    assert paths.json_path.exists()
    assert paths.csv_path.exists()


def test_benchmark_model_training_mode_tracks_memory() -> None:
    torch.manual_seed(0)
    dataset = TensorDataset(torch.randn(16, 4), torch.randn(16, 1))
    dataloader = DataLoader(dataset, batch_size=4)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    report = benchmark_model(
        model,
        dataloader,
        mode="training",
        warmup_steps=0,
        measure_steps=2,
        loss_fn=loss_fn,
        optimizer=optimizer,
        dataloader_label="train",
    )

    metrics = report.metrics
    assert metrics.latency_mean_ms > 0
    assert metrics.latency_std_ms >= 0
    if metrics.cpu_rss_delta_mb is not None:
        assert metrics.cpu_rss_delta_mb >= 0
    assert report.settings.mode == "training"
    assert report.settings.dataloader == "train"
