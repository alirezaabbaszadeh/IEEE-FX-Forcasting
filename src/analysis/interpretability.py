"""Interpretability utilities building on the instrumented MoE transformer."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

LOGGER = logging.getLogger(__name__)


@dataclass
class AttentionHeatmapResult:
    """Metadata describing a rendered attention heatmap."""

    layer_index: int
    head_index: int
    weights: np.ndarray
    figure_path: Path


@dataclass
class ExpertTraceResult:
    """Container with expert utilisation statistics and their visualisation."""

    table: pd.DataFrame
    figure_path: Optional[Path]


@dataclass
class AttributionResult:
    """Gradient attribution scores backed by an optional visualisation."""

    table: pd.DataFrame
    figure_path: Optional[Path]


@dataclass
class MarketEvent:
    """Container describing a market event to analyse."""

    event_id: str
    inputs: Tensor
    baseline: Optional[Tensor] = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    token_labels: Optional[Sequence[str]] = None
    feature_names: Optional[Sequence[str]] = None


@dataclass
class MarketEventArtifactSummary:
    """Summary of assets generated for a market event."""

    event_id: str
    output_dir: Path
    attention: List[AttentionHeatmapResult]
    expert_trace: ExpertTraceResult
    attribution: AttributionResult
    metadata: Mapping[str, object]


def _default_token_labels(length: int) -> List[str]:
    return [f"t{idx}" for idx in range(length)]


def _move_to_device(data: object, device: torch.device) -> object:
    if isinstance(data, Tensor):
        return data.to(device)
    if isinstance(data, Mapping):
        return type(data)((key, _move_to_device(value, device)) for key, value in data.items())
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return type(data)(_move_to_device(value, device) for value in data)
    return data


def _collect_attention_weights(
    model: nn.Module,
    inputs: Tensor,
    device: Optional[torch.device] = None,
) -> Dict[int, List[np.ndarray]]:
    if not hasattr(model, "register_attention_hook"):
        raise AttributeError("Model does not expose a 'register_attention_hook' method")

    attention_store: MutableMapping[int, List[np.ndarray]] = {}
    device = device or next(model.parameters()).device

    reset = getattr(model, "reset_expert_statistics", None)
    if callable(reset):
        reset()

    def hook(weights: Tensor, context: Mapping[str, object]) -> None:
        layer_idx = int(context.get("layer_index", len(attention_store)))
        head_weights = weights.detach().to("cpu")
        if head_weights.ndim == 4:
            # (batch, heads, query, key)
            aggregated = head_weights.mean(dim=0)
        elif head_weights.ndim == 3:
            aggregated = head_weights
        else:  # pragma: no cover - defensive branch for exotic shapes
            raise ValueError(f"Unexpected attention weight shape: {tuple(head_weights.shape)}")
        attention_store.setdefault(layer_idx, []).append(aggregated.numpy())

    handles = model.register_attention_hook(hook)
    handles = handles if isinstance(handles, Sequence) else [handles]

    try:
        model.eval()
        with torch.no_grad():
            model(_move_to_device(inputs, device))
    finally:
        for handle in handles:
            handle.remove()

    return {layer_idx: values for layer_idx, values in attention_store.items()}


def generate_attention_heatmaps(
    model: nn.Module,
    inputs: Tensor,
    output_dir: Path,
    *,
    token_labels: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
) -> List[AttentionHeatmapResult]:
    """Run a forward pass and render attention heatmaps for each head."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = _collect_attention_weights(model, inputs, device)

    results: List[AttentionHeatmapResult] = []
    for layer_idx, matrices in weights.items():
        stacked = np.stack(matrices, axis=0)  # (samples, heads, query, key)
        averaged = stacked.mean(axis=0)
        num_heads = averaged.shape[0]
        seq_len = averaged.shape[1]
        labels = list(token_labels) if token_labels is not None else _default_token_labels(seq_len)
        for head_idx in range(num_heads):
            matrix = averaged[head_idx]
            fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
            im = ax.imshow(matrix, cmap="viridis", aspect="auto")
            ax.set_title(f"Layer {layer_idx} · Head {head_idx}")
            ax.set_xlabel("Key")
            ax.set_ylabel("Query")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            path = output_dir / f"attention_layer{layer_idx:02d}_head{head_idx:02d}.svg"
            fig.tight_layout()
            fig.savefig(path)
            plt.close(fig)
            results.append(
                AttentionHeatmapResult(
                    layer_index=layer_idx,
                    head_index=head_idx,
                    weights=matrix,
                    figure_path=path,
                )
            )
    return results


def generate_expert_utilization_trace(
    model: nn.Module,
    output_dir: Optional[Path] = None,
    *,
    prefix: str = "expert_utilization",
) -> ExpertTraceResult:
    """Convert expert activation summaries into a long-form dataframe."""

    if not hasattr(model, "expert_activation_summaries"):
        raise AttributeError("Model does not expose 'expert_activation_summaries'")

    summaries = model.expert_activation_summaries()
    records: List[Dict[str, float]] = []
    for layer_idx, summary in enumerate(summaries):
        mean_activation = summary.get("mean_activation", [])
        top_frequency = summary.get("top_frequency", [])
        for expert_idx, (mean_val, freq_val) in enumerate(zip(mean_activation, top_frequency)):
            records.append(
                {
                    "layer": layer_idx,
                    "expert": expert_idx,
                    "mean_activation": float(mean_val),
                    "top_frequency": float(freq_val),
                }
            )
    table = pd.DataFrame.from_records(records)

    figure_path: Optional[Path] = None
    if output_dir is not None and not table.empty:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_path = output_dir / f"{prefix}.svg"
        fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=150, sharey=True)
        for metric, ax in zip(["mean_activation", "top_frequency"], axes):
            pivot = table.pivot(index="expert", columns="layer", values=metric)
            im = ax.imshow(pivot.values, cmap="magma", aspect="auto")
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("Layer")
            ax.set_ylabel("Expert")
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels(pivot.index)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(figure_path)
        plt.close(fig)

    return ExpertTraceResult(table=table, figure_path=figure_path)


def compute_gradient_attributions(
    model: nn.Module,
    inputs: Tensor,
    *,
    baseline: Optional[Tensor] = None,
    target_index: int = 0,
    steps: int = 32,
    feature_names: Optional[Sequence[str]] = None,
    output_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> AttributionResult:
    """Compute integrated gradients style attributions for model inputs."""

    model.eval()
    device = device or next(model.parameters()).device
    inputs = inputs.to(device)
    baseline = baseline.to(device) if baseline is not None else torch.zeros_like(inputs)
    diff = inputs - baseline

    total_grad = torch.zeros_like(inputs)
    alphas = torch.linspace(0.0, 1.0, steps, device=device)

    for alpha in alphas:
        blended = (baseline + alpha * diff).detach().requires_grad_(True)
        outputs = model(blended)
        target = outputs[..., target_index]
        grads = torch.autograd.grad(target.sum(), blended, retain_graph=False)[0]
        total_grad += grads

    attributions = (diff * total_grad) / steps
    attributions = attributions.detach().cpu().numpy()

    dims = attributions.shape
    feature_labels = (
        list(feature_names) if feature_names is not None else [f"f{idx}" for idx in range(dims[-1])]
    )

    records: List[Dict[str, object]] = []
    if attributions.ndim == 2:
        # (batch, features)
        for sample_idx, sample in enumerate(attributions):
            for feature_idx, value in enumerate(sample):
                records.append(
                    {
                        "sample": sample_idx,
                        "time": 0,
                        "feature": feature_labels[feature_idx],
                        "attribution": float(value),
                    }
                )
    elif attributions.ndim == 3:
        # (batch, time, features)
        for sample_idx, sample in enumerate(attributions):
            for time_idx, timestep in enumerate(sample):
                for feature_idx, value in enumerate(timestep):
                    records.append(
                        {
                            "sample": sample_idx,
                            "time": time_idx,
                            "feature": feature_labels[feature_idx],
                            "attribution": float(value),
                        }
                    )
    else:  # pragma: no cover - defensive fallback
        raise ValueError(f"Unsupported attribution tensor shape: {dims}")

    table = pd.DataFrame.from_records(records)

    figure_path: Optional[Path] = None
    if output_path is not None and not table.empty:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mean_attrib = table.groupby(["time", "feature"])  # type: ignore[arg-type]
        matrix = mean_attrib["attribution"].mean().unstack(fill_value=0.0)
        fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
        im = ax.imshow(matrix.values, cmap="coolwarm", aspect="auto")
        ax.set_title("Mean attribution")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Time")
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        figure_path = output_path

    return AttributionResult(table=table, figure_path=figure_path)


def _slugify(value: object) -> str:
    return str(value).replace(" ", "_").replace("/", "_").lower()


def analyse_market_events(
    model: nn.Module,
    events: Sequence[MarketEvent],
    output_root: Path,
    *,
    device: Optional[torch.device] = None,
) -> Tuple[List[MarketEventArtifactSummary], pd.DataFrame]:
    """Generate interpretability artefacts for a batch of events."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results: List[MarketEventArtifactSummary] = []
    metadata_records: List[Dict[str, object]] = []

    for index, event in enumerate(events):
        event_dir = output_root / _slugify(event.event_id or f"event-{index}")
        attention_dir = event_dir / "attention"
        experts_dir = event_dir / "experts"
        attrib_dir = event_dir / "attributions"

        heatmaps = generate_attention_heatmaps(
            model,
            event.inputs,
            attention_dir,
            token_labels=event.token_labels,
            device=device,
        )

        trace = generate_expert_utilization_trace(model, experts_dir)
        experts_dir.mkdir(parents=True, exist_ok=True)
        trace_csv = experts_dir / "utilization.csv"
        trace.table.to_csv(trace_csv, index=False)

        attribution = compute_gradient_attributions(
            model,
            event.inputs,
            baseline=event.baseline,
            feature_names=event.feature_names,
            output_path=attrib_dir / "mean_attribution.svg",
            device=device,
        )
        attribution.table.to_csv(attrib_dir / "attributions.csv", index=False)

        result = MarketEventArtifactSummary(
            event_id=event.event_id,
            output_dir=event_dir,
            attention=heatmaps,
            expert_trace=trace,
            attribution=attribution,
            metadata=event.metadata,
        )
        results.append(result)

        metadata_records.append(
            {
                "event_index": index,
                "event_id": event.event_id,
                "attention_count": len(heatmaps),
                "attention_dir": str(attention_dir),
                "expert_trace_csv": str(trace_csv),
                "expert_trace_figure": str(trace.figure_path) if trace.figure_path else "",
                "attribution_csv": str(attrib_dir / "attributions.csv"),
                "attribution_figure": str(attribution.figure_path) if attribution.figure_path else "",
                **{f"meta_{key}": value for key, value in (event.metadata or {}).items()},
            }
        )

    metadata = pd.DataFrame.from_records(metadata_records)
    return results, metadata


def _build_demo_model(input_dim: int, seq_len: int) -> Tuple[nn.Module, Tensor]:
    from src.models.moe_transformer import MoETransformerConfig, MoETransformerModel

    cfg = MoETransformerConfig(
        input_dim=input_dim,
        model_dim=16,
        num_layers=2,
        num_heads=2,
        moe_hidden_dim=32,
        num_experts=3,
        dropout=0.1,
        output_dim=1,
    )
    model = MoETransformerModel(cfg)
    inputs = torch.randn(4, seq_len, input_dim)
    return model, inputs


def main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover - CLI behaviour
    parser = argparse.ArgumentParser(description="Run interpretability demo artefacts.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/interpretability"))
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--features", type=int, default=6)
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model, inputs = _build_demo_model(args.features, args.seq_len)
    LOGGER.info("Generating attention heatmaps at %s", output_dir)
    heatmaps = generate_attention_heatmaps(model, inputs, output_dir / "attention")
    LOGGER.info("Rendered %d attention heatmaps", len(heatmaps))

    trace = generate_expert_utilization_trace(model, output_dir / "experts")
    trace.table.to_csv(output_dir / "experts" / "utilization.csv", index=False)

    attribution = compute_gradient_attributions(
        model,
        inputs,
        output_path=output_dir / "attributions" / "mean_attribution.svg",
    )
    attribution.table.to_csv(output_dir / "attributions" / "attributions.csv", index=False)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO)
    main()
