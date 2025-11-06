"""Compact LSTM baseline that reuses the shared scaling pipeline."""

from __future__ import annotations

from dataclasses import dataclass

try:  # pragma: no cover - optional dependency guard
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - fallback when torch is absent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class LightLSTMConfig:
    """Configuration for the lightweight LSTM baseline."""

    input_features: int
    time_steps: int
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    bidirectional: bool = False


if nn is not None:  # pragma: no branch - executed when torch is available

    class LightLSTMModel(nn.Module):
        """A minimal LSTM forecaster with a single projection head."""

        def __init__(self, config: LightLSTMConfig):
            super().__init__()
            self.config = config
            dropout = config.dropout if config.num_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=config.input_features,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=config.bidirectional,
            )
            direction_multiplier = 2 if config.bidirectional else 1
            self.dropout = nn.Dropout(config.dropout)
            self.projection = nn.Linear(config.hidden_size * direction_multiplier, 1)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs, _ = self.lstm(inputs)
            last_timestep = outputs[:, -1, :]
            last_timestep = self.dropout(last_timestep)
            prediction = self.projection(last_timestep)
            return prediction

else:  # pragma: no cover - placeholder when torch is unavailable

    class LightLSTMModel:  # type: ignore[override]
        def __init__(self, config: LightLSTMConfig):
            raise ImportError("PyTorch is required to instantiate LightLSTMModel")


__all__ = ["LightLSTMConfig", "LightLSTMModel"]
