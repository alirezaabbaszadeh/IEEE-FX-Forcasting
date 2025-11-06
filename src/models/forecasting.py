"""PyTorch forecasting models derived from the stable TensorFlow prototype in `v_10`."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence

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
    model_name: str = "temporal_transformer"
    hidden_size: int = 128
    conv_kernel_size: int = 3
    num_blocks: int = 2
    attention_heads: int = 4
    lstm_hidden_size: int = 256
    dropout: float = 0.1


if nn is not None:  # pragma: no branch - executed when torch is available

    class _HookHandle:
        """Lightweight handle mirroring the interface of PyTorch hooks."""

        def __init__(
            self,
            hooks: List[Callable[[torch.Tensor, Mapping[str, object]], None]],
            hook: Callable[[torch.Tensor, Mapping[str, object]], None],
        ) -> None:
            self._hooks = hooks
            self._hook = hook

        def remove(self) -> None:
            if self._hook in self._hooks:
                self._hooks.remove(self._hook)

    class AttentionHookMixin:
        """Mixin that enables external consumers to register attention hooks."""

        def __init__(self) -> None:
            self._attention_hooks: List[Callable[[torch.Tensor, Mapping[str, object]], None]] = []

        def register_attention_hook(
            self, hook: Callable[[torch.Tensor, Mapping[str, object]], None]
        ) -> _HookHandle:
            self._attention_hooks.append(hook)
            return _HookHandle(self._attention_hooks, hook)

        def _dispatch_attention_hooks(
            self, weights: torch.Tensor, context: Mapping[str, object]
        ) -> None:
            for hook in list(self._attention_hooks):
                hook(weights, context)

    class HeadActivationLogger:
        """Approximates expert statistics by treating attention heads as experts."""

        def __init__(self, num_heads: int) -> None:
            self.num_heads = num_heads
            self.reset()

        def reset(self) -> None:
            self._activation_sum = torch.zeros(self.num_heads)
            self._top_counts = torch.zeros(self.num_heads)
            self._entropy_sum = 0.0
            self._observations = 0

        def log(self, attn_weights: torch.Tensor) -> None:
            with torch.no_grad():
                weights = attn_weights.detach()
                if weights.ndim == 4:
                    per_head = weights.mean(dim=(0, 2, 3))
                    head_scores = weights.mean(dim=(2, 3))
                elif weights.ndim == 3:
                    per_head = weights.mean(dim=(0, 1))
                    head_scores = weights.mean(dim=1)
                else:  # pragma: no cover - defensive guard
                    raise ValueError(f"Unexpected attention weight shape: {tuple(weights.shape)}")

                normalised = head_scores / head_scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                entropy = -(normalised * normalised.clamp_min(1e-8).log()).sum(dim=-1)
                top_indices = head_scores.argmax(dim=-1).reshape(-1)
                counts = torch.bincount(top_indices.cpu(), minlength=self.num_heads)

                self._activation_sum += per_head.cpu()
                self._top_counts += counts
                self._entropy_sum += float(entropy.mean().item())
                self._observations += 1

        def summary(self) -> Dict[str, Sequence[float]]:
            if self._observations == 0:
                return {
                    "mean_activation": [0.0] * self.num_heads,
                    "top_frequency": [0.0] * self.num_heads,
                    "mean_entropy": [0.0],
                }

            mean_activation = (self._activation_sum / self._observations).tolist()
            total = float(self._top_counts.sum().item()) or 1.0
            top_frequency = (self._top_counts / total).tolist()
            entropy = self._entropy_sum / max(self._observations, 1)
            return {
                "mean_activation": [float(x) for x in mean_activation],
                "top_frequency": [float(x) for x in top_frequency],
                "mean_entropy": [float(entropy)],
            }

    class ResidualTemporalBlock(AttentionHookMixin, nn.Module):
        """A lightweight residual block mixing convolutions and self-attention."""

        def __init__(
            self, hidden_size: int, kernel_size: int, attention_heads: int, dropout: float
        ) -> None:
            AttentionHookMixin.__init__(self)
            nn.Module.__init__(self)
            padding = kernel_size // 2
            self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=padding)
            self.norm1 = nn.BatchNorm1d(hidden_size)
            self.attention = nn.MultiheadAttention(
                hidden_size, num_heads=attention_heads, dropout=dropout, batch_first=False
            )
            self.norm2 = nn.LayerNorm(hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.LeakyReLU(0.03)
            self.head_logger = HeadActivationLogger(attention_heads)

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
            attn_out, attn_weights = self.attention(
                attn_input,
                attn_input,
                attn_input,
                need_weights=True,
                average_attn_weights=False,
            )
            self._dispatch_attention_hooks(attn_weights, {"stage": "self_attention"})
            self.head_logger.log(attn_weights.detach())
            attn_out = attn_out.transpose(0, 1)
            attn_out = self.norm2(attn_out)
            attn_out = self.dropout(attn_out)

            return self.activation(attn_out + residual)

        def reset_statistics(self) -> None:
            self.head_logger.reset()

        def expert_summary(self) -> Dict[str, Sequence[float]]:
            return self.head_logger.summary()

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

        def register_attention_hook(
            self, hook: Callable[[torch.Tensor, Mapping[str, object]], None]
        ) -> List[_HookHandle]:
            """Register the provided hook on each residual block."""

            handles: List[_HookHandle] = []
            for idx, block in enumerate(self.blocks):
                if isinstance(block, AttentionHookMixin):

                    def wrapped(
                        weights: torch.Tensor,
                        context: Mapping[str, object],
                        *,
                        layer_index: int = idx,
                    ) -> None:
                        hook(weights, {**context, "layer_index": layer_index})

                    handles.append(block.register_attention_hook(wrapped))
            return handles

        def expert_activation_summaries(self) -> List[Dict[str, Sequence[float]]]:
            """Return attention-head utilisation statistics for each block."""

            summaries: List[Dict[str, Sequence[float]]] = []
            for block in self.blocks:
                if isinstance(block, ResidualTemporalBlock):
                    summaries.append(block.expert_summary())
            return summaries

        def reset_expert_statistics(self) -> None:
            for block in self.blocks:
                if isinstance(block, ResidualTemporalBlock):
                    block.reset_statistics()

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
