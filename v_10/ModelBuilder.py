# /no_csv_1/10/ModelBuilder.py

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Flatten,
    LeakyReLU, BatchNormalization, AdditiveAttention, Add, MultiHeadAttention
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW
from typing import List, Dict, Any, Optional

# Set a global seed for reproducibility, a critical step for academic publications.
SEED = 42
tf.random.set_seed(SEED)

class MixOfExperts(tf.keras.layers.Layer):
    """
    Custom Mix of Experts (MoE) Layer.

    Enhances model capacity by using specialized expert networks. A gating network
    learns to combine the outputs of these experts, allowing the model to handle
    diverse and complex patterns in the data.

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
        self.experts = [Dense(units, activation=LeakyReLU(alpha=self.alpha)) for _ in range(num_experts)]
        self.gate = Dense(num_experts, activation='linear')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the MoE layer."""
        gate_values = self.gate(inputs)
        expert_outputs = tf.stack([expert(inputs) for expert in self.experts], axis=-1)
        gated_output = tf.reduce_sum(expert_outputs * tf.expand_dims(gate_values, -2), axis=-1)
        return gated_output
    
    def get_config(self):
        """Returns the serializable configuration of the layer."""
        config = super(MixOfExperts, self).get_config()
        config.update({"num_experts": self.num_experts, "units": self.units, "alpha": self.alpha})
        return config

class ModelBuilder:
    """
    Constructs the time series forecasting model for Version 10.

    This architecture integrates several advanced components for robust feature learning:
    1.  **Residual Blocks with Multi-Head Attention:** For capturing local and complex patterns.
    2.  **Bidirectional LSTMs:** To understand long-range temporal dependencies.
    3.  **Additive Attention:** To weigh the significance of different time steps from LSTM outputs.
    4.  **Mix of Experts (MoE):** To increase model capacity and allow for specialization.
    """

    def __init__(self,
                 time_steps: int,
                 num_features: int,
                 block_configs: List[Dict[str, Any]],
                 # Residual Block MHA Parameters
                 num_heads: int = 3,
                 key_dim: int = 4,
                 leaky_relu_alpha_res_block_1: float = 0.04,
                 leaky_relu_alpha_res_block_2: float = 0.03,
                 leaky_relu_alpha_after_add: float = 0.03,
                 conv_l2_reg: float = 0.0,
                 # LSTM Parameters
                 lstm_units: int = 200,
                 recurrent_dropout_lstm: float = 0.3,
                 lstm_l2_reg: float = 0.0,
                 # MoE Parameters
                 moe_num_experts: int = 32,
                 moe_units: int = 64,
                 moe_leaky_relu_alpha: float = 0.01,
                 # Optimizer Parameters
                 optimizer_lr: float = 0.001,
                 optimizer_clipnorm: float = None,
                 **kwargs):
        """Initializes the ModelBuilder for Version 10 with a comprehensive set of hyperparameters."""
        self.time_steps = time_steps
        self.num_features = num_features
        self.block_configs = block_configs
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.leaky_relu_alpha_res_block_1 = leaky_relu_alpha_res_block_1
        self.leaky_relu_alpha_res_block_2 = leaky_relu_alpha_res_block_2
        self.leaky_relu_alpha_after_add = leaky_relu_alpha_after_add
        self.conv_kernel_regularizer = l2(conv_l2_reg) if conv_l2_reg > 0 else None
        self.lstm_units = lstm_units
        self.recurrent_dropout_lstm = recurrent_dropout_lstm
        self.lstm_kernel_regularizer = l2(lstm_l2_reg) if lstm_l2_reg > 0 else None
        self.moe_num_experts = moe_num_experts
        self.moe_units = moe_units
        self.moe_leaky_relu_alpha = moe_leaky_relu_alpha
        self.optimizer_lr = optimizer_lr
        self.optimizer_clipnorm = optimizer_clipnorm

    def _residual_block(self, x: tf.Tensor, filters: int, kernel_size: int, pool_size: Optional[int], block_num: int) -> tf.Tensor:
        """Constructs a residual block for Version 10 using Multi-Head Attention."""
        shortcut = x
        
        # First convolutional path with MHA
        x_path = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', name=f'res_block{block_num}_conv1', kernel_regularizer=self.conv_kernel_regularizer)(x)
        x_path = BatchNormalization(name=f'res_block{block_num}_bn1')(x_path)
        x_path = LeakyReLU(alpha=self.leaky_relu_alpha_res_block_1, name=f'res_block{block_num}_relu1')(x_path)
        x_path = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, name=f'res_block{block_num}_mha1')(x_path, x_path)
        x_path = BatchNormalization(name=f'res_block{block_num}_bn_after_mha1')(x_path)
        if pool_size:
            x_path = MaxPooling1D(pool_size=pool_size, padding='same')(x_path)

        # Second convolutional path with MHA
        x_path = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', name=f'res_block{block_num}_conv2', kernel_regularizer=self.conv_kernel_regularizer)(x_path)
        x_path = BatchNormalization(name=f'res_block{block_num}_bn2')(x_path)
        x_path = LeakyReLU(alpha=self.leaky_relu_alpha_res_block_2, name=f'res_block{block_num}_relu2')(x_path)
        x_path = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, name=f'res_block{block_num}_mha2')(x_path, x_path)
        x_path = BatchNormalization(name=f'res_block{block_num}_bn_after_mha2')(x_path)
        if pool_size:
            x_path = MaxPooling1D(pool_size=pool_size, padding='same')(x_path)

        # Shortcut connection
        if shortcut.shape[-1] != filters or shortcut.shape[-2] != x_path.shape[-2]:
            shortcut = Conv1D(filters=filters, kernel_size=1, padding='same', name=f'res_block{block_num}_shortcut_conv')(shortcut)
            shortcut = BatchNormalization(name=f'res_block{block_num}_shortcut_bn')(shortcut)
            if pool_size and x_path.shape[-2] != shortcut.shape[-2]:
                total_pool = pool_size * pool_size
                shortcut = MaxPooling1D(pool_size=total_pool, padding='same')(shortcut)
            
        x_add = Add(name=f'res_block{block_num}_add')([shortcut, x_path])
        return LeakyReLU(alpha=self.leaky_relu_alpha_after_add, name=f'res_block{block_num}_relu_after_add')(x_add)

    def build_model(self) -> Model:
        """Builds and compiles the complete Keras model for Version 10."""
        input_layer = Input(shape=(self.time_steps, self.num_features), dtype='float32', name='input_layer')
        x = input_layer

        for i, block_config in enumerate(self.block_configs):
            x = self._residual_block(x, **block_config, block_num=i)

        x = Bidirectional(LSTM(self.lstm_units, return_sequences=True, recurrent_dropout=self.recurrent_dropout_lstm, kernel_regularizer=self.lstm_kernel_regularizer, dtype='float32'))(x)
        # x = BatchNormalization(name='bn_after_lstm')(x)

        x = AdditiveAttention(name='final_additive_attention')([x, x])
        x = BatchNormalization(name='bn_after_attention')(x)
        x = MixOfExperts(num_experts=self.moe_num_experts, units=self.moe_units, alpha=self.moe_leaky_relu_alpha, name='moe_layer')(x)
        x = BatchNormalization(name='bn_after_moe')(x)

        x = Flatten(name='flatten_layer')(x)
        output = Dense(1, activation='linear', name='output_layer', dtype='float32')(x)

        model = Model(inputs=input_layer, outputs=output, name='TimeSeriesForecasting_V10')
        optimizer = AdamW(learning_rate=self.optimizer_lr)
        # clipnorm=self.optimizer_clipnorm
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        return model