import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Flatten,
    Dropout, # Kept for potential future use, though default is off
    LeakyReLU,
    BatchNormalization,
    MultiHeadAttention,
    Add
    # ReLU # Not used in V6's LeakyReLU-based architecture
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW
from typing import List, Dict, Any, Optional

# Global seed for TensorFlow operations (recommended to be set in Run.py for script-level reproducibility)
SEED = 42
tf.random.set_seed(SEED)

# Custom Layer: MixOfExperts (Copied from V7/V8 ModelBuilder for completeness, though V6 does not use it)
# This is included to ensure no import errors if other parts of the system expect it,
# but it is NOT part of the V6 architecture itself.
class MixOfExperts(tf.keras.layers.Layer):
    def __init__(self, num_experts: int = 32, units: int = 64, alpha: float = 0.01, **kwargs):
        super(MixOfExperts, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.units = units
        self.alpha = alpha
        self.experts = [Dense(units, name=f'expert_{i}') for i in range(num_experts)]
        self.expert_activations = [LeakyReLU(alpha=self.alpha, name=f'expert_leaky_relu_{i}') for i in range(num_experts)]
        self.gate = Dense(num_experts, activation='linear', name='gate_linear')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        gate_values = self.gate(inputs)
        expert_outputs_list = []
        for i in range(self.num_experts):
            expert_out = self.experts[i](inputs)
            expert_out = self.expert_activations[i](expert_out)
            expert_outputs_list.append(expert_out)
        expert_outputs_stacked = tf.stack(expert_outputs_list, axis=-1)
        gate_values_expanded = tf.expand_dims(gate_values, axis=2)
        weighted_expert_outputs = expert_outputs_stacked * gate_values_expanded
        gated_output = tf.reduce_sum(weighted_expert_outputs, axis=-1)
        return gated_output
        
    def get_config(self):
        config = super(MixOfExperts, self).get_config()
        config.update({
            "num_experts": self.num_experts,
            "units": self.units,
            "alpha": self.alpha,
        })
        return config

class ModelBuilder:
    """
    Builds a time series forecasting model with Version 6 architecture:
    Input -> [Configurable Residual Blocks with Conv1D & MHA] -> BiLSTM_Block ->
    MultiHeadAttention (with V6-specific head/key_dim calculation) -> Flatten -> Dense -> Output.

    Hyperparameters are configurable via the constructor and are expected to be
    standardized based on the V8 template.
    """

    def __init__(self,
                 time_steps: int,
                 num_features: int,
                 block_configs: List[Dict[str, Any]],
                 # Residual Block MHA parameters
                 num_heads_res_block: int = 12,
                 key_dim_res_block: int = 4,
                 # LeakyReLU alphas for Residual Block
                 leaky_relu_alpha_conv1_res_block: float = 0.04,
                 leaky_relu_alpha_conv2_res_block: float = 0.03, # For 2nd conv in res_block
                 leaky_relu_alpha_after_add_res_block: float = 0.03, # For LeakyRelu after Add in res_block
                 conv_l2_reg_res_block: float = 0.0,
                 
                 # LSTM parameters
                 num_bilstm_layers: int = 1,
                 lstm_units: int = 200,
                 recurrent_dropout_lstm: float = 0.0, # Standardized to 0.0
                 lstm_l2_reg: float = 0.0,
                 use_batchnorm_after_lstm: bool = True,

                 # Parameters for the MultiHeadAttention layer AFTER LSTM
                 # These are derived inside build_model using num_heads_res_block and key_dim_res_block
                 use_batchnorm_after_post_lstm_mha: bool = True,

                 # Dense layer after Flatten (before output)
                 dense_units_after_flatten: int = 256,
                 leaky_relu_alpha_dense_after_flatten: float = 0.00,
                 dense_l2_reg_after_flatten: float = 0.0,
                 
                 # Output Layer
                 output_activation: str = 'linear', # Typically 'linear' for regression
                 output_l2_reg: float = 0.0,
                 
                 # Optimizer
                 optimizer_lr: float = 0.01,
                 optimizer_weight_decay: Optional[float] = None,
                 optimizer_clipnorm: Optional[float] = None,
                 optimizer_clipvalue: Optional[float] = None,
                 **kwargs): # To catch any unused params from MainClass

        self.time_steps = time_steps
        self.num_features = num_features
        self.block_configs = block_configs

        # Store hyperparameters for residual blocks
        self.num_heads_res_block = num_heads_res_block
        self.key_dim_res_block = key_dim_res_block
        self.leaky_relu_alpha_conv1_res_block = leaky_relu_alpha_conv1_res_block
        self.leaky_relu_alpha_conv2_res_block = leaky_relu_alpha_conv2_res_block
        self.leaky_relu_alpha_after_add_res_block = leaky_relu_alpha_after_add_res_block
        self.conv_kernel_regularizer_res_block = l2(conv_l2_reg_res_block) if conv_l2_reg_res_block > 0 else None

        # Store hyperparameters for LSTM layers
        self.num_bilstm_layers = num_bilstm_layers
        self.lstm_units = lstm_units
        self.recurrent_dropout_lstm = recurrent_dropout_lstm # Expected to be 0.0
        self.lstm_kernel_regularizer = l2(lstm_l2_reg) if lstm_l2_reg > 0 else None
        self.use_batchnorm_after_lstm = use_batchnorm_after_lstm

        # Store hyperparameters for post-LSTM MHA
        self.use_batchnorm_after_post_lstm_mha = use_batchnorm_after_post_lstm_mha
        # Actual num_heads and key_dim for post-LSTM MHA are calculated in build_model

        # Store hyperparameters for the Dense layer after flatten
        self.dense_units_after_flatten = dense_units_after_flatten
        self.leaky_relu_alpha_dense_after_flatten = leaky_relu_alpha_dense_after_flatten
        self.dense_kernel_regularizer_after_flatten = l2(dense_l2_reg_after_flatten) if dense_l2_reg_after_flatten > 0 else None
        
        # Store hyperparameters for the output layer
        self.output_activation = output_activation
        self.output_kernel_regularizer = l2(output_l2_reg) if output_l2_reg > 0 else None
        
        # Store optimizer parameters
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.optimizer_clipnorm = optimizer_clipnorm
        self.optimizer_clipvalue = optimizer_clipvalue

    def _residual_block(self, x: tf.Tensor, filters: int, kernel_size: int, pool_size: Optional[int], block_num: int) -> tf.Tensor:
        """
        Constructs a residual block for Version 6.
        Each block: Conv1D->BN->LeakyReLU -> MHA->BN -> Conv1D->BN->LeakyReLU -> MHA->BN, then Add shortcut.
        Pooling is applied after each MHA if pool_size is specified.
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
        x_bn_after_mha1 = BatchNormalization(name=f'resblock{block_num}_bn_after_mha1')(x_mha1)
        
        current_path = x_bn_after_mha1
        if pool_size is not None and pool_size > 1:
            current_path = MaxPooling1D(pool_size=pool_size, padding='same', name=f'resblock{block_num}_pool1')(current_path)

        # Second Conv-MHA sequence
        x_conv2 = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', 
                         kernel_regularizer=self.conv_kernel_regularizer_res_block, name=f'resblock{block_num}_conv2')(current_path)
        x_bn2 = BatchNormalization(name=f'resblock{block_num}_bn2')(x_conv2)
        x_relu2 = LeakyReLU(alpha=self.leaky_relu_alpha_conv2_res_block, name=f'resblock{block_num}_relu2')(x_bn2)

        mha2 = MultiHeadAttention(num_heads=self.num_heads_res_block, key_dim=self.key_dim_res_block,
                                  name=f'resblock{block_num}_mha2')
        x_mha2 = mha2(query=x_relu2, value=x_relu2, key=x_relu2)
        x_bn_after_mha2 = BatchNormalization(name=f'resblock{block_num}_bn_after_mha2')(x_mha2)
        
        current_path = x_bn_after_mha2
        if pool_size is not None and pool_size > 1: # Second pooling in the block
            current_path = MaxPooling1D(pool_size=pool_size, padding='same', name=f'resblock{block_num}_pool2')(current_path)
        
        # Shortcut connection
        if shortcut.shape[-1] != filters or shortcut.shape[-2] != current_path.shape[-2]:
            shortcut = Conv1D(filters=filters, kernel_size=1, padding='same', 
                              kernel_regularizer=self.conv_kernel_regularizer_res_block, name=f'resblock{block_num}_shortcut_conv')(shortcut)
            shortcut = BatchNormalization(name=f'resblock{block_num}_shortcut_bn')(shortcut)
            # If pooling was applied in the main path, apply equivalent pooling to shortcut
            if x.shape[-2] != current_path.shape[-2] and pool_size is not None and pool_size > 1:
                # This simplified pooling logic assumes if current_path was pooled, shortcut needs similar reduction.
                # This matches the original V6's pooling structure where it was applied twice if pool_size was set.
                shortcut = MaxPooling1D(pool_size=pool_size, padding='same', name=f'resblock{block_num}_shortcut_pool1')(shortcut)
                if MaxPooling1D(pool_size=pool_size, padding='same')(shortcut).shape[-2] == current_path.shape[-2]: # Check if a second pool is needed
                     shortcut = MaxPooling1D(pool_size=pool_size, padding='same', name=f'resblock{block_num}_shortcut_pool2')(shortcut)


        added = Add(name=f'resblock{block_num}_add')([shortcut, current_path])
        output = LeakyReLU(alpha=self.leaky_relu_alpha_after_add_res_block, name=f'resblock{block_num}_relu_after_add')(added)
        return output

    def build_model(self) -> Model:
        """
        Constructs and compiles the Keras model for Version 6.
        """
        input_layer = Input(shape=(self.time_steps, self.num_features), dtype='float32', name='input_layer')
        x = input_layer

        # Residual Blocks
        for i, block_conf in enumerate(self.block_configs):
            filters = block_conf.get('filters', 8) # Default from V8 standard
            kernel_size = block_conf.get('kernel_size', 3) # Default from V8 standard
            pool_size = block_conf.get('pool_size', None) # Default from V8 standard (was 2 in original V6 Run)
            x = self._residual_block(x, filters, kernel_size, pool_size, block_num=i)

        # Bidirectional LSTM Layers
        for i in range(self.num_bilstm_layers):
            x = Bidirectional(LSTM(self.lstm_units,
                                   return_sequences=True,
                                   recurrent_dropout=self.recurrent_dropout_lstm, # Should be 0.0
                                   kernel_regularizer=self.lstm_kernel_regularizer,
                                   dtype='float32'), name=f'bilstm_layer_{i+1}')(x)
            if self.use_batchnorm_after_lstm:
                x = BatchNormalization(name=f'bn_after_bilstm_{i+1}')(x)
        
        # Post-LSTM MultiHeadAttention (V6 specific calculation for heads/key_dim)
        post_lstm_mha_num_heads = self.num_heads_res_block + len(self.block_configs)
        post_lstm_mha_key_dim = self.key_dim_res_block # Original V6: self.key_dim * 1
        
        attention_layer_post_lstm = MultiHeadAttention(
            num_heads=post_lstm_mha_num_heads,
            key_dim=post_lstm_mha_key_dim,
            name='post_lstm_mha'
        )
        x = attention_layer_post_lstm(query=x, value=x, key=x)
        if self.use_batchnorm_after_post_lstm_mha:
            x = BatchNormalization(name='bn_after_post_lstm_mha')(x)

        # Flatten and Dense layers
        x = Flatten(name='flatten_layer')(x)
        
        x = Dense(self.dense_units_after_flatten, 
                  kernel_regularizer=self.dense_kernel_regularizer_after_flatten,
                  name='dense_after_flatten')(x)
        x = BatchNormalization(name='bn_after_dense_flatten')(x) # BN was present in original V6
        x = LeakyReLU(alpha=self.leaky_relu_alpha_dense_after_flatten, 
                      name='leaky_relu_after_dense_flatten')(x)
        
        # Output Layer
        output = Dense(1, activation=self.output_activation, 
                       kernel_regularizer=self.output_kernel_regularizer,
                       dtype='float32', name='output_layer')(x)

        model = Model(inputs=input_layer, outputs=output, name='TimeSeriesForecastingModel_V6')

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
            # Post-LSTM MHA heads/key_dim are derived, not stored directly as separate HPs

            "dense_units_after_flatten": self.dense_units_after_flatten,
            "leaky_relu_alpha_dense_after_flatten": self.leaky_relu_alpha_dense_after_flatten,
            "dense_l2_reg_after_flatten": self.dense_kernel_regularizer_after_flatten.l2 if self.dense_kernel_regularizer_after_flatten else 0.0,
            
            "output_activation": self.output_activation,
            "output_l2_reg": self.output_kernel_regularizer.l2 if self.output_kernel_regularizer else 0.0,
            
            "optimizer_lr": self.optimizer_lr,
            "optimizer_weight_decay": self.optimizer_weight_decay,
            "optimizer_clipnorm": self.optimizer_clipnorm,
            "optimizer_clipvalue": self.optimizer_clipvalue,
        }
        # Note: MixOfExperts is not part of V6, so its params are not in get_config
        return config
