"""Transformer-style forecasting model with Mix-of-Experts support.

This module refactors the TensorFlow prototype located in ``v_10`` into a
PyTorch implementation with a modular design.  The building blocks expose
attention hooks so downstream experiments can inspect attention weights, and
the Mix-of-Experts (MoE) layer records activation statistics for later
analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn


class _HookHandle:
    """Lightweight handle that mimics the behaviour of PyTorch hook handles."""

    def __init__(self, hooks: List[Callable[[Tensor, Mapping[str, object]], None]], hook: Callable[[Tensor, Mapping[str, object]], None]) -> None:
        self._hooks = hooks
        self._hook = hook

    def remove(self) -> None:
        if self._hook in self._hooks:
            self._hooks.remove(self._hook)


class AttentionHookMixin:
    """Mixin that allows registration of callables receiving attention weights."""

    def __init__(self) -> None:
        self._attention_hooks: List[Callable[[Tensor, Mapping[str, object]], None]] = []

    def register_attention_hook(
        self, hook: Callable[[Tensor, Mapping[str, object]], None]
    ) -> _HookHandle:
        """Register a callable that receives attention weights during forward passes."""

        self._attention_hooks.append(hook)
        return _HookHandle(self._attention_hooks, hook)

    def _dispatch_attention_hooks(self, weights: Tensor, context: Mapping[str, object]) -> None:
        for hook in list(self._attention_hooks):
            hook(weights, context)


class ExpertActivationLogger:
    """Aggregates gating statistics for a Mix-of-Experts layer."""

    def __init__(self, num_experts: int) -> None:
        self.num_experts = num_experts
        self.reset()

    def reset(self) -> None:
        self._activation_sum = torch.zeros(self.num_experts)
        self._top_counts = torch.zeros(self.num_experts)
        self._observations = 0

    def log(self, gate_probs: Tensor) -> None:
        with torch.no_grad():
            probs = gate_probs.detach()
            if probs.ndim == 2:  # (batch, expert)
                mean_per_expert = probs.mean(dim=0)
            elif probs.ndim >= 3:
                dims = tuple(range(probs.ndim - 1))
                mean_per_expert = probs.mean(dim=dims)
            else:  # pragma: no cover - defensive
                raise ValueError("Gate probabilities must have at least 2 dimensions")

            self._activation_sum += mean_per_expert.cpu()
            top_indices = probs.argmax(dim=-1).reshape(-1)
            counts = torch.bincount(top_indices.cpu(), minlength=self.num_experts)
            self._top_counts += counts
            self._observations += 1

    def summary(self) -> Dict[str, Sequence[float]]:
        if self._observations == 0:
            return {
                "mean_activation": [0.0] * self.num_experts,
                "top_frequency": [0.0] * self.num_experts,
            }

        mean_activation = (self._activation_sum / self._observations).tolist()
        top_total = float(self._top_counts.sum().item()) or 1.0
        top_frequency = (self._top_counts / top_total).tolist()
        return {
            "mean_activation": [float(x) for x in mean_activation],
            "top_frequency": [float(x) for x in top_frequency],
        }


class MoELayer(nn.Module):
    """Simple feed-forward Mix-of-Experts layer with gating logger."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int,
        activation: Optional[nn.Module] = None,
        logger: Optional[ExpertActivationLogger] = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.experts = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)
        self.activation = activation or nn.LeakyReLU(0.01)
        self.logger = logger or ExpertActivationLogger(num_experts)

    def forward(self, inputs: Tensor) -> Tensor:  # pragma: no cover - exercised in integration tests
        gate_logits = self.gate(inputs)
        gate_probs = gate_logits.softmax(dim=-1)
        expert_outputs = torch.stack([self.activation(expert(inputs)) for expert in self.experts], dim=-1)
        self.logger.log(gate_probs)
        weighted = (expert_outputs * gate_probs.unsqueeze(-2)).sum(dim=-1)
        return weighted


class ResidualMoEBlock(AttentionHookMixin, nn.Module):
    """Residual block combining multi-head attention with a Mix-of-Experts feed-forward."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        moe_hidden_dim: int,
        num_experts: int,
        dropout: float = 0.1,
    ) -> None:
        AttentionHookMixin.__init__(self)
        nn.Module.__init__(self)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.moe = MoELayer(embed_dim, moe_hidden_dim, num_experts)
        self.moe_proj = nn.Linear(moe_hidden_dim, embed_dim)
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs: Tensor) -> Tensor:  # pragma: no cover - exercised in integration tests
        attn_out, attn_weights = self.self_attn(inputs, inputs, inputs, need_weights=True, average_attn_weights=False)
        self._dispatch_attention_hooks(attn_weights, {"stage": "self_attention"})
        x = self.attn_norm(inputs + self.attn_dropout(attn_out))

        moe_out = self.moe(x)
        moe_projected = self.moe_proj(moe_out)
        x = self.ff_norm(x + self.ff_dropout(moe_projected))
        return x


@dataclass
class MoETransformerConfig:
    """Configuration for :class:`MoETransformerModel`."""

    input_dim: int
    model_dim: int
    num_layers: int
    num_heads: int
    moe_hidden_dim: int
    num_experts: int
    dropout: float = 0.1
    output_dim: int = 1


class MoETransformerModel(nn.Module):
    """Encoder-only transformer with Mix-of-Experts feed-forward layers."""

    def __init__(self, cfg: MoETransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_projection = nn.Linear(cfg.input_dim, cfg.model_dim)
        self.layers = nn.ModuleList(
            [
                ResidualMoEBlock(
                    embed_dim=cfg.model_dim,
                    num_heads=cfg.num_heads,
                    moe_hidden_dim=cfg.moe_hidden_dim,
                    num_experts=cfg.num_experts,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.model_dim)
        self.head = nn.Linear(cfg.model_dim, cfg.output_dim)

    def register_attention_hook(
        self, hook: Callable[[Tensor, Mapping[str, object]], None]
    ) -> List[_HookHandle]:
        """Register the same hook for each transformer block."""

        handles = []
        for idx, layer in enumerate(self.layers):
            def wrapped(weights: Tensor, context: Mapping[str, object], *, layer_index: int = idx) -> None:
                hook(weights, {**context, "layer_index": layer_index})

            handles.append(layer.register_attention_hook(wrapped))
        return handles

    def expert_activation_summaries(self) -> List[Dict[str, Sequence[float]]]:
        summaries: List[Dict[str, Sequence[float]]] = []
        for layer in self.layers:
            summaries.append(layer.moe.logger.summary())
        return summaries

    def forward(self, inputs: Tensor) -> Tensor:  # pragma: no cover - exercised in integration tests
        x = self.input_projection(inputs)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)

