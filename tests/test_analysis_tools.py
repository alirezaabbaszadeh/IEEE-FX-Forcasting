from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.analysis import (
    BenchmarkReport,
    benchmark_model,
    compute_gradient_attributions,
    generate_attention_heatmaps,
    generate_expert_utilization_trace,
    save_report,
)
from src.models.moe_transformer import MoETransformerConfig, MoETransformerModel
from scripts.export_figures import export_figures
from scripts.export_tables import export_tables


def _build_demo_model() -> MoETransformerModel:
    cfg = MoETransformerConfig(
        input_dim=4,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        moe_hidden_dim=16,
        num_experts=2,
        dropout=0.1,
        output_dim=1,
    )
    return MoETransformerModel(cfg)


def test_interpretability_suite(tmp_path: Path) -> None:
    model = _build_demo_model()
    inputs = torch.randn(3, 6, 4)

    heatmaps = generate_attention_heatmaps(model, inputs, tmp_path / "attention")
    assert heatmaps, "Expected at least one attention heatmap"
    seq_len = inputs.shape[1]
    for result in heatmaps:
        assert result.figure_path.exists()
        assert result.weights.shape[0] == seq_len

    trace = generate_expert_utilization_trace(model, tmp_path / "experts")
    assert not trace.table.empty
    assert {"layer", "expert", "mean_activation", "top_frequency"} <= set(trace.table.columns)
    assert trace.figure_path is not None and trace.figure_path.exists()

    attributions = compute_gradient_attributions(
        model,
        inputs,
        output_path=tmp_path / "attrib" / "mean.svg",
    )
    assert not attributions.table.empty
    assert attributions.figure_path is not None and attributions.figure_path.exists()


def test_benchmark_and_reporting(tmp_path: Path) -> None:
    dataset = TensorDataset(torch.randn(32, 4), torch.randn(32, 1))
    dataloader = DataLoader(dataset, batch_size=8)
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    report = benchmark_model(
        model,
        dataloader,
        mode="training",
        warmup_steps=1,
        measure_steps=2,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )
    assert isinstance(report, BenchmarkReport)
    assert report.metrics.throughput_samples_per_sec > 0
    assert report.hardware.device

    report_path = save_report(report, tmp_path / "benchmarks")
    assert report_path.exists()

    metrics_path = Path("artifacts/examples/metrics.csv")
    outputs = export_tables([metrics_path], tmp_path / "tables")
    for path in outputs.values():
        assert path.exists()
        if path.suffix == ".csv":
            pd.read_csv(path)

    figures = export_figures([metrics_path], tmp_path / "figures")
    assert figures
    for figure in figures:
        assert figure.exists()
