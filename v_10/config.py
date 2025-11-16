"""Configuration dataclasses for the Version 10 pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DataParameters:
    """Configuration values required to prepare and split the dataset."""

    file_path: str
    time_steps: int = 3
    train_ratio: float = 0.96
    val_ratio: float = 0.02
    test_ratio: float = 0.02


@dataclass
class TrainingParameters:
    """Configuration for the model training phase."""

    epochs: int = 60
    batch_size: int = 5000
    early_stopping_patience: int = 20
    reduce_lr_patience: int = 1
    reduce_lr_factor: float = 0.1
    min_lr: float = 5e-7


@dataclass
class ModelBuilderConfig:
    """Hyperparameters that govern model construction and compilation."""

    block_configs: List[Dict[str, Any]] = field(
        default_factory=lambda: [{'filters': 8, 'kernel_size': 3, 'pool_size': None}]
    )
    num_heads: int = 3
    key_dim: int = 4
    leaky_relu_alpha_res_block_1: float = 0.04
    leaky_relu_alpha_res_block_2: float = 0.03
    leaky_relu_alpha_after_add: float = 0.03
    conv_l2_reg: float = 0.0
    lstm_units: int = 200
    recurrent_dropout_lstm: float = 0.0
    lstm_l2_reg: float = 0.0
    moe_num_experts: int = 12
    moe_units: int = 64
    moe_leaky_relu_alpha: float = 0.01
    optimizer_lr: float = 0.01
    optimizer_clipnorm: Optional[float] = None


@dataclass
class PipelineConfig:
    """Top-level container with all configuration required for a pipeline run."""

    data: DataParameters
    training: TrainingParameters
    model_builder: ModelBuilderConfig
    base_dir: str
