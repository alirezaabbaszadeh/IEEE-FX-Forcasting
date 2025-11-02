from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pandas as pd
import torch

from src.cli import run_interpret_command


def test_run_interpret_command_generates_assets(tmp_path, monkeypatch):
    module_code = dedent(
        """
        import torch
        from torch import nn

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 1)
                self._hooks = []
                self._attention_weights = torch.full((1, 2, 4, 4), 0.25)
                with torch.no_grad():
                    self.linear.weight[:] = torch.tensor([[0.2, 0.3, 0.5]])
                    self.linear.bias.zero_()

            def register_attention_hook(self, hook):
                self._hooks.append(hook)

                model = self

                class Handle:
                    def remove(self_inner):
                        if hook in model._hooks:
                            model._hooks.remove(hook)

                return Handle()

            def expert_activation_summaries(self):
                return [
                    {
                        "mean_activation": [0.4, 0.6],
                        "top_frequency": [0.5, 0.5],
                        "mean_entropy": [0.69],
                    }
                ]

            def reset_expert_statistics(self):
                pass

            def forward(self, inputs):
                weights = self._attention_weights.expand(
                    inputs.shape[0], -1, inputs.shape[1], inputs.shape[1]
                )
                for hook in list(self._hooks):
                    hook(weights, {"layer_index": 0})
                pooled = inputs.mean(dim=1)
                return self.linear(pooled)

        def build_model():
            return DummyModel()
        """
    )

    module_path = tmp_path / "dummy_model.py"
    module_path.write_text(module_code)
    monkeypatch.syspath_prepend(str(tmp_path))

    events = [
        {
            "event_id": "event_a",
            "inputs": torch.randn(2, 4, 3),
            "metadata": {"pair": "EURUSD", "horizon": "1h", "success": True},
            "feature_names": ["f1", "f2", "f3"],
        },
        {
            "event_id": "event_b",
            "inputs": torch.randn(1, 4, 3),
            "metadata": {"pair": "EURUSD", "horizon": "1h", "success": False},
            "feature_names": ["f1", "f2", "f3"],
        },
    ]
    events_path = tmp_path / "events.pt"
    torch.save(events, events_path)

    output_dir = tmp_path / "artifacts"
    metadata_path = run_interpret_command(
        model_module="dummy_model",
        model_factory="build_model",
        events_path=events_path,
        output_dir=output_dir,
        seed=42,
    )

    assert metadata_path.exists()
    expected_root = output_dir / "eurusd" / "1h" / "seed-42"
    assert metadata_path.parent == expected_root
    metadata = pd.read_csv(metadata_path)
    assert set(["event_a", "event_b"]).issubset(set(metadata["event_id"]))
    assert (metadata["event_count"] == 2).all()
    assert (metadata["seed"] == 42).all()
    assert (metadata["pair"] == "EURUSD").all()
    assert (metadata["horizon"] == "1h").all()
    expected_columns = {
        "event_index",
        "event_id",
        "attention_count",
        "attention_dir",
        "expert_trace_csv",
        "expert_trace_figure",
        "attribution_csv",
        "attribution_figure",
        "gating_entropy_csv",
        "annotation_csv",
        "annotation_json",
        "success",
        "meta_pair",
        "meta_horizon",
        "meta_success",
        "event_count",
        "seed",
        "model_module",
        "model_factory",
        "pair",
        "horizon",
    }
    assert expected_columns <= set(metadata.columns)

    base_dir = metadata_path.parent
    for event in ["event_a", "event_b"]:
        event_dir = base_dir / event
        attention_files = sorted((event_dir / "attention").glob("*.svg"))
        assert attention_files, "Expected attention heatmaps"
        expert_csv = event_dir / "experts" / "utilization.csv"
        assert expert_csv.exists()
        expert_table = pd.read_csv(expert_csv)
        assert {
            "layer",
            "expert",
            "mean_activation",
            "top_frequency",
            "mean_entropy",
        } <= set(expert_table.columns)
        gating_path = event_dir / "experts" / "gating_entropy_correlations.csv"
        assert gating_path.exists()
        gating_table = pd.read_csv(gating_path)
        assert {"layer", "entropy_activation_corr", "entropy_top_frequency_corr"} <= set(
            gating_table.columns
        )
        attrib_csv = event_dir / "attributions" / "attributions.csv"
        assert attrib_csv.exists()
        attrib_table = pd.read_csv(attrib_csv)
        assert not attrib_table.empty
        assert {"sample", "feature"} <= set(attrib_table.columns)
        annotation_csv = event_dir / "annotations.csv"
        annotation_json = event_dir / "annotations.json"
        assert annotation_csv.exists()
        assert annotation_json.exists()
        annotation_table = pd.read_csv(annotation_csv)
        assert {"event_id", "success", "outcome"} <= set(annotation_table.columns)
