import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Flatten,
    Dropout, # Kept for future flexibility, though standardized to 0.0 for now
    LeakyReLU,
    BatchNormalization,
    MultiHeadAttention,
    Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW
from typing import List, Dict, Any, Optional

# Global seed for TensorFlow operations (recommended to be set in Run.py for script-level reproducibility)
SEED = 42
tf.random.set_seed(SEED)

class MixOfExperts(tf.keras.layers.Layer):
    """
    A Mix of Experts (MoE) layer.
    Consists of multiple "expert" Dense sub-networks and a "gating" network
    that learns to assign weights to the outputs of these experts.
    The final output is a weighted sum of the expert outputs.
    """
    def __init__(self, 
                 num_experts: int = 32, 
                 units: int = 64, 
                 leaky_relu_alpha: float = 0.01, # Alpha for LeakyReLU in experts
                 **kwargs):
        """
        Initializes the MixOfExperts layer.

        Args:
            num_experts (int): Number of expert networks.
            units (int): Number of output units for each expert Dense layer.
            leaky_relu_alpha (float): Alpha (negative slope) for LeakyReLU activation in experts.
        """
        super(MixOfExperts, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.units = units
        self.leaky_relu_alpha = leaky_relu_alpha

        # Define expert layers, each being a Dense layer with LeakyReLU activation.
        self.experts = [Dense(units, name=f'expert_{i}') for i in range(num_experts)]
        self.expert_activations = [LeakyReLU(alpha=self.leaky_relu_alpha, name=f'expert_leaky_relu_{i}') for i in range(num_experts)]
        
        # Gating network with linear activation (as in original V7)
        self.gate = Dense(num_experts, activation='linear', name='gate_linear')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass for the MixOfExperts layer.
        """
        gate_values = self.gate(inputs) # Shape: (batch_size, ..., num_experts)

        expert_outputs_list = []
        for i in range(self.num_experts):
            expert_out = self.experts[i](inputs) # Shape: (batch_size, ..., units)
            expert_out = self.expert_activations[i](expert_out)
            expert_outputs_list.append(expert_out)
        
        expert_outputs_stacked = tf.stack(expert_outputs_list, axis=-1) # Shape: (batch_size, ..., units, num_experts)
        gate_values_expanded = tf.expand_dims(gate_values, axis=-2) # Target: (batch_size, ..., 1, num_experts)
        
        weighted_expert_outputs = expert_outputs_stacked * gate_values_expanded
        gated_output = tf.reduce_sum(weighted_expert_outputs, axis=-1) # Shape: (batch_size, ..., units)
        
        return gated_output
        
    def get_config(self):
        config = super(MixOfExperts, self).get_config()
        config.update({
            "num_experts": self.num_experts,
            "units": self.units,
            "leaky_relu_alpha": self.leaky_relu_alpha,
        })
        return config

class ModelBuilder:
    """
    Builds a time series forecasting model with Version 7 architecture:
    Input -> [Residual Blocks (Conv1D+MHA, V7 dual pooling)] -> BiLSTM ->
    MHA (V7-specific config) -> MixOfExperts -> Flatten -> Output Dense.
    """

    def __init__(self,
                 time_steps: int,
                 num_features: int,
                 block_configs: List[Dict[str, Any]],
                 # Residual Block parameters
                 num_heads_res_block: int = 12,
                 key_dim_res_block: int = 4,
                 leaky_relu_alpha_conv1_res_block: float = 0.04,
                 leaky_relu_alpha_conv2_res_block: float = 0.03, 
                 leaky_relu_alpha_after_add_res_block: float = 0.03,
                 conv_l2_reg_res_block: float = 0.0,
                 
                 # LSTM parameters
                 num_bilstm_layers: int = 1,
                 lstm_units: int = 200,
                 recurrent_dropout_lstm: float = 0.0, # Standardized: OFF
                 lstm_l2_reg: float = 0.0,
                 use_batchnorm_after_lstm: bool = True,

                 # Post-LSTM MultiHeadAttention parameters
                 use_batchnorm_after_post_lstm_mha: bool = True,

                 # MixOfExperts parameters
                 moe_num_experts: int = 32,
                 moe_units: int = 64,
                 moe_leaky_relu_alpha: float = 0.01, 
                 use_batchnorm_after_moe: bool = True,
                 
                 # Output Layer parameters
                 output_activation: str = 'linear',
                 output_l2_reg: float = 0.0,
                 
                 # Optimizer
                 optimizer_lr: float = 0.01,
                 optimizer_weight_decay: Optional[float] = None,
                 optimizer_clipnorm: Optional[float] = None,
                 optimizer_clipvalue: Optional[float] = None,
                 **kwargs): 

        self.time_steps = time_steps
        self.num_features = num_features
        self.block_configs = block_configs

        self.num_heads_res_block = num_heads_res_block
        self.key_dim_res_block = key_dim_res_block
        self.leaky_relu_alpha_conv1_res_block = leaky_relu_alpha_conv1_res_block
        self.leaky_relu_alpha_conv2_res_block = leaky_relu_alpha_conv2_res_block
        self.leaky_relu_alpha_after_add_res_block = leaky_relu_alpha_after_add_res_block
        self.conv_kernel_regularizer_res_block = l2(conv_l2_reg_res_block) if conv_l2_reg_res_block > 0 else None

        self.num_bilstm_layers = num_bilstm_layers
        self.lstm_units = lstm_units
        self.recurrent_dropout_lstm = recurrent_dropout_lstm
        self.lstm_kernel_regularizer = l2(lstm_l2_reg) if lstm_l2_reg > 0 else None
        self.use_batchnorm_after_lstm = use_batchnorm_after_lstm

        self.use_batchnorm_after_post_lstm_mha = use_batchnorm_after_post_lstm_mha

        self.moe_num_experts = moe_num_experts
        self.moe_units = moe_units
        self.moe_leaky_relu_alpha = moe_leaky_relu_alpha
        self.use_batchnorm_after_moe = use_batchnorm_after_moe
        
        self.output_activation = output_activation
        self.output_kernel_regularizer = l2(output_l2_reg) if output_l2_reg > 0 else None
        
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.optimizer_clipnorm = optimizer_clipnorm
        self.optimizer_clipvalue = optimizer_clipvalue

    def _residual_block(self, x: tf.Tensor, filters: int, kernel_size: int, pool_size: Optional[int], block_num: int) -> tf.Tensor:
        """
        Constructs a V7-style residual block.
        Conv1D->BN->LReLU -> MHA->BN -> Pool_A? -> Conv1D->BN->LReLU -> MHA->BN -> Pool_B? -> Add -> LReLU.
        """
        shortcut = x

        # First Conv-MHA sequence
        x_conv1 = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', 
                         kernel_regularizer=self.conv_kernel_regularizer_res_block, name=f'resblock{block_num}_conv1')(x)
        x_bn1 = BatchNormalization(name=f'resblock{block_num}_bn1')(x_conv1)
        x_relu1 = LeakyReLU(alpha=self.leaky_relu_alpha_conv1_res_block, name=f'resblock{block_num}_relu1')(x_bn1)
        
        mha1 = MultiHeadAttention(num_heads=self.num_heads_res_block, key_dim=self.key_dim_res_block,
                                  name=f'resblock{block_num}_mha1')
        x_mha1 = mha1(query=x_relu1, value=x_relu1, key=x_relu1)
        current_path = BatchNormalization(name=f'resblock{block_num}_bn_after_mha1')(x_mha1)
        
        if pool_size is not None and pool_size > 1: # V7's first pooling
            current_path = MaxPooling1D(pool_size=pool_size, padding='same', name=f'resblock{block_num}_pool_A')(current_path)

        # Second Conv-MHA sequence
        x_conv2 = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', 
                         kernel_regularizer=self.conv_kernel_regularizer_res_block, name=f'resblock{block_num}_conv2')(current_path)
        x_bn2 = BatchNormalization(name=f'resblock{block_num}_bn2')(x_conv2)
        x_relu2 = LeakyReLU(alpha=self.leaky_relu_alpha_conv2_res_block, name=f'resblock{block_num}_relu2')(x_bn2)

        mha2 = MultiHeadAttention(num_heads=self.num_heads_res_block, key_dim=self.key_dim_res_block,
                                  name=f'resblock{block_num}_mha2')
        x_mha2 = mha2(query=x_relu2, value=x_relu2, key=x_relu2)
        current_path = BatchNormalization(name=f'resblock{block_num}_bn_after_mha2')(x_mha2)
        
        if pool_size is not None and pool_size > 1: # V7's second pooling
            current_path = MaxPooling1D(pool_size=pool_size, padding='same', name=f'resblock{block_num}_pool_B')(current_path)
        
        # Shortcut connection
        if shortcut.shape[-1] != filters or shortcut.shape[-2] != current_path.shape[-2]:
            shortcut = Conv1D(filters=filters, kernel_size=1, padding='same',
                              kernel_regularizer=self.conv_kernel_regularizer_res_block, 
                              name=f'resblock{block_num}_shortcut_conv')(shortcut)
            shortcut = BatchNormalization(name=f'resblock{block_num}_shortcut_bn')(shortcut)
            
            # Apply pooling to shortcut to match main path's sequence length reduction
            # This logic handles the dual pooling specific to V7's residual block.
            if x.shape[-2] != current_path.shape[-2] and pool_size is not None and pool_size > 1:
                # Determine how many pooling operations were effectively applied to current_path from original x
                # Check if shortcut needs one pool
                if shortcut.shape[-2] // pool_size == current_path.shape[-2]:
                    shortcut = MaxPooling1D(pool_size=pool_size, padding='same', name=f'resblock{block_num}_shortcut_pool_A')(shortcut)
                # Check if shortcut needs two pools
                elif shortcut.shape[-2] // (pool_size * pool_size) == current_path.shape[-2]:
                    shortcut = MaxPooling1D(pool_size=pool_size, padding='same', name=f'resblock{block_num}_shortcut_pool_A')(shortcut)
                    shortcut = MaxPooling1D(pool_size=pool_size, padding='same', name=f'resblock{block_num}_shortcut_pool_B')(shortcut)
        
        added = Add(name=f'resblock{block_num}_add')([shortcut, current_path])
        output = LeakyReLU(alpha=self.leaky_relu_alpha_after_add_res_block, name=f'resblock{block_num}_relu_after_add')(added)
        return output

    def build_model(self) -> Model:
        """
        Builds and compiles the Keras model for Version 7.
        """
        input_layer = Input(shape=(self.time_steps, self.num_features), dtype='float32', name='input_layer')
        x = input_layer

        # Residual Blocks
        for i, block_conf in enumerate(self.block_configs):
            filters = block_conf.get('filters', 8) 
            kernel_size = block_conf.get('kernel_size', 3)
            pool_size = block_conf.get('pool_size', 2) # V7's architectural choice for pooling
            x = self._residual_block(x, filters, kernel_size, pool_size, block_num=i)

        # Bidirectional LSTM Layers
        for i in range(self.num_bilstm_layers):
            x = Bidirectional(LSTM(self.lstm_units,
                                   return_sequences=True, 
                                   recurrent_dropout=self.recurrent_dropout_lstm,
                                   kernel_regularizer=self.lstm_kernel_regularizer,
                                   dtype='float32'), name=f'bilstm_layer_{i+1}')(x)
            if self.use_batchnorm_after_lstm:
                x = BatchNormalization(name=f'bn_after_bilstm_{i+1}')(x)
        
        # Post-LSTM MultiHeadAttention (V7 specific calculation for heads/key_dim)
        post_lstm_mha_num_heads = self.num_heads_res_block + len(self.block_configs)
        post_lstm_mha_key_dim = self.key_dim_res_block 
        
        attention_layer_post_lstm = MultiHeadAttention(
            num_heads=post_lstm_mha_num_heads,
            key_dim=post_lstm_mha_key_dim,
            name='post_lstm_mha'
        )
        x = attention_layer_post_lstm(query=x, value=x, key=x)
        if self.use_batchnorm_after_post_lstm_mha:
            x = BatchNormalization(name='bn_after_post_lstm_mha')(x)

        # MixOfExperts Layer
        moe_layer = MixOfExperts(
            num_experts=self.moe_num_experts,
            units=self.moe_units,
            leaky_relu_alpha=self.moe_leaky_relu_alpha,
            name='mix_of_experts'
        )
        x = moe_layer(x)
        if self.use_batchnorm_after_moe:
            x = BatchNormalization(name='bn_after_moe')(x)
        
        # Flatten Layer (No intermediate Dense layer before output in V7)
        x = Flatten(name='flatten_layer')(x)
        
        # Output Layer
        output = Dense(1, activation=self.output_activation, 
                       kernel_regularizer=self.output_kernel_regularizer,
                       dtype='float32', name='output_layer')(x)

        model = Model(inputs=input_layer, outputs=output, name='TimeSeriesForecastingModel_V7')

        # Optimizer
        optimizer_kwargs = {'learning_rate': self.optimizer_lr}
        if self.optimizer_weight_decay is not None:
            optimizer_kwargs['weight_decay'] = self.optimizer_weight_decay
        if self.optimizer_clipnorm is not None:
            optimizer_kwargs['clipnorm'] = self.optimizer_clipnorm
        if self.optimizer_clipvalue is not None:
            optimizer_kwargs['clipvalue'] = self.optimizer_clipvalue
        
        optimizer = AdamW(**optimizer_kwargs)

        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        return model

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the ModelBuilder."""
        config = {
            "time_steps": self.time_steps,
            "num_features": self.num_features,
            "block_configs": self.block_configs,
            "num_heads_res_block": self.num_heads_res_block,
            "key_dim_res_block": self.key_dim_res_block,
            "leaky_relu_alpha_conv1_res_block": self.leaky_relu_alpha_conv1_res_block,
            "leaky_relu_alpha_conv2_res_block": self.leaky_relu_alpha_conv2_res_block,
            "leaky_relu_alpha_after_add_res_block": self.leaky_relu_alpha_after_add_res_block,
            "conv_l2_reg_res_block": self.conv_kernel_regularizer_res_block.l2 if self.conv_kernel_regularizer_res_block else 0.0,
            
            "num_bilstm_layers": self.num_bilstm_layers,
            "lstm_units": self.lstm_units,
            "recurrent_dropout_lstm": self.recurrent_dropout_lstm,
            "lstm_l2_reg": self.lstm_kernel_regularizer.l2 if self.lstm_kernel_regularizer else 0.0,
            "use_batchnorm_after_lstm": self.use_batchnorm_after_lstm,

            "use_batchnorm_after_post_lstm_mha": self.use_batchnorm_after_post_lstm_mha,

            "moe_num_experts": self.moe_num_experts,
            "moe_units": self.moe_units,
            "moe_leaky_relu_alpha": self.moe_leaky_relu_alpha,
            "use_batchnorm_after_moe": self.use_batchnorm_after_moe,
            
            # Parameters for the removed intermediate dense layer are not included
            
            "output_activation": self.output_activation,
            "output_l2_reg": self.output_kernel_regularizer.l2 if self.output_kernel_regularizer else 0.0,
            
            "optimizer_lr": self.optimizer_lr,
            "optimizer_weight_decay": self.optimizer_weight_decay,
            "optimizer_clipnorm": self.optimizer_clipnorm,
            "optimizer_clipvalue": self.optimizer_clipvalue,
        }
        return config
