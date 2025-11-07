"""Regime-conditioned quantile forecaster built on top of the lightweight LSTM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

try:  # pragma: no cover - optional dependency guard
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - fallback when torch is absent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class RCQFConfig:
    """Configuration for the regime-conditioned quantile forecaster."""

    input_features: int
    time_steps: int
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    bidirectional: bool = False
    num_regimes: int = 3
    gating_hidden_size: int = 64
    quantile_levels: Sequence[float] = field(default_factory=lambda: (0.05, 0.5, 0.95))

    def __post_init__(self) -> None:
        if self.input_features <= 0:
            raise ValueError("input_features must be positive")
        if self.time_steps <= 0:
            raise ValueError("time_steps must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_regimes <= 0:
            raise ValueError("num_regimes must be positive")
        if self.gating_hidden_size <= 0:
            raise ValueError("gating_hidden_size must be positive")
        if not self.quantile_levels:
            raise ValueError("quantile_levels must contain at least one entry")
        sorted_levels = sorted(float(level) for level in self.quantile_levels)
        if any(level <= 0.0 or level >= 1.0 for level in sorted_levels):
            raise ValueError("quantile levels must lie strictly between 0 and 1")
        object.__setattr__(self, "quantile_levels", tuple(sorted_levels))


if nn is not None:  # pragma: no branch - executed when torch is available

    class RCQFModel(nn.Module):
        """Mixture-of-regimes quantile forecaster driven by an LSTM encoder."""

        def __init__(self, config: RCQFConfig):
            super().__init__()
            self.config = config
            dropout = config.dropout if config.num_layers > 1 else 0.0
            self.encoder = nn.LSTM(
                input_size=config.input_features,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=config.bidirectional,
            )
            direction_multiplier = 2 if config.bidirectional else 1
            encoded_size = config.hidden_size * direction_multiplier
            self.dropout = nn.Dropout(config.dropout)
            self.regime_head = nn.Sequential(
                nn.Linear(encoded_size, config.gating_hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.gating_hidden_size, config.num_regimes),
            )
            self.quantile_head = nn.Linear(
                encoded_size, config.num_regimes * len(config.quantile_levels)
            )
            self.register_buffer(
                "quantile_levels", torch.tensor(config.quantile_levels, dtype=torch.float32)
            )
            if 0.5 in config.quantile_levels:
                self._median_index = config.quantile_levels.index(0.5)
            else:
                self._median_index = len(config.quantile_levels) // 2

        def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs, _ = self.encoder(inputs)
            last_timestep = outputs[:, -1, :]
            return self.dropout(last_timestep)

        def _compute_regime_mixture(
            self, representation: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            logits = self.regime_head(representation)
            probabilities = torch.softmax(logits, dim=-1)
            quantile_raw = self.quantile_head(representation)
            num_quantiles = len(self.config.quantile_levels)
            per_regime = quantile_raw.view(-1, self.config.num_regimes, num_quantiles)
            mixture = torch.einsum("brq,br->bq", per_regime, probabilities)
            return mixture, per_regime, probabilities, logits

        def forward(  # type: ignore[override]
            self, inputs: torch.Tensor, *, return_dict: bool = False
        ) -> torch.Tensor | dict[str, torch.Tensor]:
            representation = self._encode(inputs)
            mixture, per_regime, probabilities, logits = self._compute_regime_mixture(
                representation
            )
            median = mixture[:, self._median_index].unsqueeze(-1)
            if return_dict:
                return {
                    "median": median,
                    "quantiles": mixture,
                    "regime_probabilities": probabilities,
                    "regime_logits": logits,
                    "per_regime_quantiles": per_regime,
                    "quantile_levels": self.quantile_levels,
                }
            return median

else:  # pragma: no cover - placeholder when torch is unavailable

    class RCQFModel:  # type: ignore[override]
        def __init__(self, config: RCQFConfig):
            raise ImportError("PyTorch is required to instantiate RCQFModel")


__all__ = ["RCQFConfig", "RCQFModel"]
