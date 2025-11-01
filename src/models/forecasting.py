"""PyTorch forecasting models derived from the stable TensorFlow prototype in `v_10`."""
from __future__ import annotations

import math
from dataclasses import dataclass

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - fallback for lightweight environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class ModelConfig:
    """Hyperparameters controlling :class:`TemporalForecastingModel`."""

    input_features: int
    time_steps: int
    hidden_size: int = 128
    conv_kernel_size: int = 3
    num_blocks: int = 2
    attention_heads: int = 4
    lstm_hidden_size: int = 256
    dropout: float = 0.1


if nn is not None:  # pragma: no branch - executed when torch is available

    class ResidualTemporalBlock(nn.Module):
        """A lightweight residual block mixing convolutions and self-attention."""

        def __init__(self, hidden_size: int, kernel_size: int, attention_heads: int, dropout: float) -> None:
            super().__init__()
            padding = kernel_size // 2
            self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=padding)
            self.norm1 = nn.BatchNorm1d(hidden_size)
            self.attention = nn.MultiheadAttention(
                hidden_size, num_heads=attention_heads, dropout=dropout, batch_first=False
            )
            self.norm2 = nn.LayerNorm(hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.LeakyReLU(0.03)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            # inputs: (batch, time, hidden)
            residual = inputs
            conv_in = inputs.transpose(1, 2)
            conv_out = self.conv(conv_in)
            conv_out = self.norm1(conv_out)
            conv_out = self.activation(conv_out)
            conv_out = self.dropout(conv_out)
            conv_out = conv_out.transpose(1, 2)

            attn_input = conv_out.transpose(0, 1)  # (time, batch, hidden)
            attn_out, _ = self.attention(attn_input, attn_input, attn_input)
            attn_out = attn_out.transpose(0, 1)
            attn_out = self.norm2(attn_out)
            attn_out = self.dropout(attn_out)

            return self.activation(attn_out + residual)


    class TemporalForecastingModel(nn.Module):
        """End-to-end forecasting network inspired by the `v_10` TensorFlow architecture."""

        def __init__(self, cfg: ModelConfig) -> None:
            super().__init__()
            self.cfg = cfg

            self.input_proj = nn.Linear(cfg.input_features, cfg.hidden_size)
            self.blocks = nn.ModuleList(
                [
                    ResidualTemporalBlock(
                        hidden_size=cfg.hidden_size,
                        kernel_size=cfg.conv_kernel_size,
                        attention_heads=cfg.attention_heads,
                        dropout=cfg.dropout,
                    )
                    for _ in range(cfg.num_blocks)
                ]
            )

            self.lstm = nn.LSTM(
                input_size=cfg.hidden_size,
                hidden_size=cfg.lstm_hidden_size,
                batch_first=True,
                bidirectional=True,
            )
            self.dropout = nn.Dropout(cfg.dropout)
            self.head = nn.Sequential(
                nn.Linear(cfg.lstm_hidden_size * 2, cfg.lstm_hidden_size),
                nn.LeakyReLU(0.03),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.lstm_hidden_size, 1),
            )

            self._reset_parameters()

        def _reset_parameters(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Conv1d):
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
            x = self.input_proj(inputs)
            for block in self.blocks:
                x = block(x)
            lstm_out, _ = self.lstm(x)
            pooled = lstm_out[:, -1, :]
            pooled = self.dropout(pooled)
            return self.head(pooled)


else:  # pragma: no cover - executed when torch is unavailable

    class TemporalForecastingModel:  # type: ignore[override]
        """Placeholder implementation signalling that PyTorch is required."""

        def __init__(self, cfg: ModelConfig) -> None:  # pragma: no cover - simple guard
            raise ImportError("PyTorch is required to instantiate TemporalForecastingModel")


    class ResidualTemporalBlock:  # type: ignore[override]
        def __init__(self, *args: object, **kwargs: object) -> None:  # pragma: no cover
            raise ImportError("PyTorch is required to instantiate ResidualTemporalBlock")
