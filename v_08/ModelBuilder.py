# /no_csv_1/8/ModelBuilder.py

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Flatten,
    LeakyReLU, BatchNormalization, MultiHeadAttention, Add, ReLU
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW
from typing import List, Dict, Any, Optional

# Set a global seed for reproducibility, a critical step for academic research.
SEED = 42
tf.random.set_seed(SEED)

class MixOfExperts(tf.keras.layers.Layer):
    """
    Custom Mix of Experts (MoE) Layer.

    This layer enhances model capacity by using specialized expert networks. A gating
    network learns how to combine the outputs of these experts, allowing the model
    to handle diverse and complex patterns in the data.

    Attributes:
        num_experts (int): The number of expert sub-networks.
        units (int): The output dimensionality of each expert.
        alpha (float): The negative slope for the LeakyReLU activation.
    """
    def __init__(self, num_experts: int = 32, units: int = 64, alpha: float = 0.01, **kwargs):
        super(MixOfExperts, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.units = units
        self.alpha = alpha
        self.experts = [Dense(units, name=f'expert_{i}') for i in range(num_experts)]
        self.expert_activations = [LeakyReLU(alpha=self.alpha, name=f'expert_leaky_relu_{i}') for i in range(num_experts)]
        self.gate = Dense(num_experts, activation='linear', name='gating_network')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass for the MoE layer."""
        gate_values = self.gate(inputs)
        expert_outputs_list = [self.expert_activations[i](self.experts[i](inputs)) for i in range(self.num_experts)]
        expert_outputs_stacked = tf.stack(expert_outputs_list, axis=-1)
        gate_values_expanded = tf.expand_dims(gate_values, axis=-2)
        weighted_expert_outputs = expert_outputs_stacked * gate_values_expanded
        gated_output = tf.reduce_sum(weighted_expert_outputs, axis=-1)
        return gated_output
        
    def get_config(self):
        """Returns the serializable configuration of the layer."""
        config = super(MixOfExperts, self).get_config()
        config.update({
            "num_experts": self.num_experts, "units": self.units, "alpha": self.alpha,
        })
        return config

class ModelBuilder:
    """
    Builds the sophisticated time series forecasting model for Version 8.

    This architecture is designed for high performance, incorporating:
    1.  A backbone of residual blocks with Conv1D and Multi-Head Attention.
    2.  Bidirectional LSTMs for long-range temporal dependency modeling.
    3.  A subsequent Multi-Head Attention layer to refine LSTM outputs.
    4.  A Mix of Experts (MoE) layer to increase model capacity.
    5.  Final Dense layers for regression output.
    """

    def __init__(self,
                 time_steps: int,
                 num_features: int,
                 block_configs: List[Dict[str, Any]],
                 num_heads_res_block: int = 12,
                 key_dim_res_block: int = 4,
                 lstm_units: int = 200,
                 num_lstm_layers: int = 1,
                 attention_after_lstm_heads: int = 12,
                 attention_after_lstm_key_dim: int = 50,
                 moe_num_experts: int = 32,
                 moe_units: int = 64,
                 dense_units_before_output: int = 256,
                 leaky_relu_alpha_dense: float = 0.00,
                 leaky_relu_alpha_res_block: float = 0.04,
                 leaky_relu_alpha_res_block2: float = 0.03,
                 l2_reg: float = 0.0,
                 optimizer_lr: float = 0.001,
                 optimizer_clipnorm: Optional[float] = None,
                 **kwargs):
        """
        Initializes the ModelBuilder with all hyperparameters for Version 8.
        """
        self.time_steps = time_steps
        self.num_features = num_features
        self.block_configs = block_configs
        self.num_heads_res_block = num_heads_res_block
        self.key_dim_res_block = key_dim_res_block
        self.lstm_units = lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.attention_after_lstm_heads = attention_after_lstm_heads
        self.attention_after_lstm_key_dim = attention_after_lstm_key_dim
        self.moe_num_experts = moe_num_experts
        self.moe_units = moe_units
        self.dense_units_before_output = dense_units_before_output
        self.leaky_relu_alpha_dense = leaky_relu_alpha_dense
        self.leaky_relu_alpha_res_block = leaky_relu_alpha_res_block
        self.leaky_relu_alpha_res_block2 = leaky_relu_alpha_res_block2
        self.l2_reg = l2_reg
        self.optimizer_lr = optimizer_lr
        self.optimizer_clipnorm = optimizer_clipnorm

    def _residual_block(self, x, filters, kernel_size, pool_size, block_num):
        """Constructs a residual block with Conv1D and MultiHeadAttention."""
        shortcut = x
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', name=f'res_block{block_num}_conv1')(x)
        x = BatchNormalization(name=f'res_block{block_num}_bn1')(x)
        x = LeakyReLU(alpha=self.leaky_relu_alpha_res_block, name=f'res_block{block_num}_leaky_relu1')(x)
        x = MultiHeadAttention(num_heads=self.num_heads_res_block, key_dim=self.key_dim_res_block, name=f'res_block{block_num}_mha1')(query=x, value=x, key=x)
        x = BatchNormalization(name=f'res_block{block_num}_bn_after_mha1')(x)
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', name=f'res_block{block_num}_conv2')(x)
        x = BatchNormalization(name=f'res_block{block_num}_bn2')(x)
        x = LeakyReLU(alpha=self.leaky_relu_alpha_res_block2, name=f'res_block{block_num}_leaky_relu2')(x)
        x = MultiHeadAttention(num_heads=self.num_heads_res_block, key_dim=self.key_dim_res_block, name=f'res_block{block_num}_mha2')(query=x, value=x, key=x)
        x = BatchNormalization(name=f'res_block{block_num}_bn_after_mha2')(x)
        if shortcut.shape[-1] != filters:
            shortcut = Conv1D(filters=filters, kernel_size=1, padding='same', name=f'res_block{block_num}_shortcut_conv')(shortcut)
            shortcut = BatchNormalization(name=f'res_block{block_num}_shortcut_bn')(shortcut)
        x = Add(name=f'res_block{block_num}_add_shortcut')([shortcut, x])
        x = LeakyReLU(alpha=self.leaky_relu_alpha_res_block2, name=f'res_block{block_num}_leaky_relu3_after_add')(x)
        return x

    def build_model(self) -> Model:
        """Builds, compiles, and returns the complete Keras model for Version 8."""
        
        # --- Input Layer with explicit float32 for stability ---
        input_layer = Input(shape=(self.time_steps, self.num_features), dtype='float32', name='input_layer')
        x = input_layer

        # --- Convolutional Backbone ---
        for i, block_config in enumerate(self.block_configs):
            x = self._residual_block(x, filters=block_config.get('filters', 8), kernel_size=block_config.get('kernel_size', 3), pool_size=block_config.get('pool_size'), block_num=i)

        # --- Recurrent Core with explicit float32 ---
        for _ in range(self.num_lstm_layers):
            x = Bidirectional(LSTM(self.lstm_units, return_sequences=True, dtype='float32'))(x)

        # --- Attention and Expert Layers ---
        x = MultiHeadAttention(num_heads=self.attention_after_lstm_heads, key_dim=self.attention_after_lstm_key_dim, name='mha_after_lstm')(query=x, value=x, key=x)
        x = BatchNormalization(name='bn_after_mha_lstm')(x)
        x = MixOfExperts(num_experts=self.moe_num_experts, units=self.moe_units, name='mix_of_experts')(x)
        x = BatchNormalization(name='bn_after_moe')(x)

        # --- Output Head with explicit float32 ---
        x = Flatten(name='flatten_layer')(x)
        x = Dense(self.dense_units_before_output, name='dense_before_output')(x)
        x = LeakyReLU(alpha=self.leaky_relu_alpha_dense, name='leaky_relu_after_dense')(x)
        output_layer = Dense(1, activation='linear', kernel_regularizer=l2(self.l2_reg) if self.l2_reg > 0 else None, dtype='float32', name='output_layer')(x)
        
        model = Model(inputs=input_layer, outputs=output_layer, name='TimeSeriesForecasting_V8')
        
        # --- Optimizer with Gradient Clipping ---
        optimizer = AdamW(learning_rate=self.optimizer_lr, clipnorm=self.optimizer_clipnorm)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        return model