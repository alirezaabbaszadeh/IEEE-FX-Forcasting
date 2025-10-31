# ModelBuilder.py for Version 5 (no csv/5/)
# Defines the neural network architecture for time series forecasting.
# This version features configurable residual blocks (defined by block_configs)
# with internal Conv1D and MultiHeadAttention layers, followed by BiLSTMs,
# a final MultiHeadAttention layer, an optional intermediate Dense layer,
# and a Dense output layer. Hyperparameters are managed externally.

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Flatten,
    Dropout,
    LeakyReLU,
    BatchNormalization,
    MultiHeadAttention,
    Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW
from typing import List, Dict, Any, Optional

# Optional: Global seed for TensorFlow operations (better set in Run.py).
SEED = 42
tf.random.set_seed(SEED)

class ModelBuilder:
    """
    Builds a time series forecasting model with Version 5 architecture:
    Input -> [Configurable Residual Blocks with Conv1D & MHA] -> BiLSTM_Block ->
    MultiHeadAttention -> Optional Intermediate Dense -> Dense_Output.

    The convolutional part is defined by `block_configs`. Other components'
    hyperparameters are configurable via the constructor. Specific layers (Input, LSTM, Output Dense)
    are set to use 'float32' dtype. Dropout and recurrent_dropout default to off.
    """

    def __init__(self,
                 time_steps: int,
                 num_features: int,
                 block_configs: List[Dict[str, Any]],
                 # Parameters for components within each residual block
                 num_heads_residual_block: int = 12,
                 key_dim_residual_block: int = 4,
                 leaky_relu_alpha_conv_1: float = 0.04,
                 leaky_relu_alpha_conv_2: float = 0.03,
                 leaky_relu_alpha_after_residual_add: float = 0.03,
                 conv_l2_reg: float = 0.0,
                 # BiLSTM Block Parameters
                 num_bilstm_layers: int = 1,
                 lstm_units: int = 200,
                 recurrent_dropout_lstm: float = 0.0, # Defaulting to 0.0 (off)
                 lstm_l2_reg: float = 0.0,
                 use_batchnorm_after_lstm: bool = False,
                 # Final MultiHeadAttention Layer (after LSTMs) Parameters
                 num_heads_final_mha: int = 12, # Default based on original V5 logic
                 key_dim_final_mha: int = 4,    # Default based on original V5 logic
                 use_batchnorm_after_final_mha: bool = True,
                 # Optional Intermediate Dense Layer Parameters
                 use_intermediate_dense: bool = False, # Defaulting to False (off)
                 intermediate_dense_units: int = 256,
                 leaky_relu_alpha_intermediate_dense: float = 0.00, # Original V5 had 0.0 for this
                 intermediate_dense_l2_reg: float = 0.0,
                 use_batchnorm_intermediate_dense: bool = True, # Original V5 had BN here
                 # Flatten and Dense Output Parameters
                 use_dropout_before_output: bool = False, # Defaulting to False (off)
                 dropout_rate_before_output: float = 0.0, # Rate is 0.0 if dropout is off
                 output_activation: str = 'linear',
                 output_l2_reg: float = 0.0,
                 # Optimizer Parameters
                 optimizer_lr: float = 0.01,
                 optimizer_weight_decay: Optional[float] = None,
                 optimizer_clipnorm: Optional[float] = None,
                 optimizer_clipvalue: Optional[float] = None
                 ):
        """
        Initializes the ModelBuilder for Version 5.
        Args are detailed in the Canvas plan for Version 5, Phase 2.
        """
        self.time_steps = time_steps
        self.num_features = num_features
        self.block_configs = block_configs

        self.num_heads_residual_block = num_heads_residual_block
        self.key_dim_residual_block = key_dim_residual_block
        self.leaky_relu_alpha_conv_1 = leaky_relu_alpha_conv_1
        self.leaky_relu_alpha_conv_2 = leaky_relu_alpha_conv_2
        self.leaky_relu_alpha_after_residual_add = leaky_relu_alpha_after_residual_add
        self.conv_kernel_regularizer = l2(conv_l2_reg) if conv_l2_reg > 0 else None

        self.num_bilstm_layers = num_bilstm_layers
        self.lstm_units = lstm_units
        self.recurrent_dropout_lstm = recurrent_dropout_lstm
        self.lstm_kernel_regularizer = l2(lstm_l2_reg) if lstm_l2_reg > 0 else None
        self.use_batchnorm_after_lstm = use_batchnorm_after_lstm

        self.num_heads_final_mha = num_heads_final_mha
        self.key_dim_final_mha = key_dim_final_mha
        self.use_batchnorm_after_final_mha = use_batchnorm_after_final_mha

        self.use_intermediate_dense = use_intermediate_dense
        self.intermediate_dense_units = intermediate_dense_units
        self.leaky_relu_alpha_intermediate_dense = leaky_relu_alpha_intermediate_dense
        self.intermediate_dense_kernel_regularizer = l2(intermediate_dense_l2_reg) if intermediate_dense_l2_reg > 0 else None
        self.use_batchnorm_intermediate_dense = use_batchnorm_intermediate_dense

        self.use_dropout_before_output = use_dropout_before_output
        self.dropout_rate_before_output = dropout_rate_before_output if self.use_dropout_before_output else 0.0
        self.output_activation = output_activation
        self.output_kernel_regularizer = l2(output_l2_reg) if output_l2_reg > 0 else None

        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.optimizer_clipnorm = optimizer_clipnorm
        self.optimizer_clipvalue = optimizer_clipvalue

    def _residual_block(self,
                       x_input: tf.Tensor,
                       filters: int,
                       kernel_size: int,
                       pool_size: Optional[int],
                       block_num: int) -> tf.Tensor:
        """
        Constructs a residual block as defined in Version 5's architecture.
        Each block contains two Conv1D-MHA sequences and a residual connection.
        Pooling is applied after each MHA sequence if pool_size is specified.
        """
        shortcut = x_input
        x = x_input

        # First Conv1D -> BN -> LeakyReLU -> MHA -> BN sequence
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same',
                   kernel_regularizer=self.conv_kernel_regularizer, name=f'resblock{block_num}_conv1a')(x)
        x = BatchNormalization(name=f'resblock{block_num}_bn1a')(x)
        x = LeakyReLU(negative_slope=self.leaky_relu_alpha_conv_1, name=f'resblock{block_num}_relu1a')(x)

        mha1 = MultiHeadAttention(num_heads=self.num_heads_residual_block, key_dim=self.key_dim_residual_block,
                                  name=f'resblock{block_num}_mha1a')
        x = mha1(query=x, value=x, key=x)
        x = BatchNormalization(name=f'resblock{block_num}_bn_after_mha1a')(x)

        if pool_size is not None and pool_size > 1: # First pooling in the block
            x = MaxPooling1D(pool_size=pool_size, padding='same', name=f'resblock{block_num}_pool1')(x)

        # Second Conv1D -> BN -> LeakyReLU -> MHA -> BN sequence
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same',
                   kernel_regularizer=self.conv_kernel_regularizer, name=f'resblock{block_num}_conv1b')(x)
        x = BatchNormalization(name=f'resblock{block_num}_bn1b')(x)
        x = LeakyReLU(negative_slope=self.leaky_relu_alpha_conv_2, name=f'resblock{block_num}_relu1b')(x)

        mha2 = MultiHeadAttention(num_heads=self.num_heads_residual_block, key_dim=self.key_dim_residual_block,
                                  name=f'resblock{block_num}_mha1b')
        x = mha2(query=x, value=x, key=x)
        x = BatchNormalization(name=f'resblock{block_num}_bn_after_mha1b')(x)

        if pool_size is not None and pool_size > 1: # Second pooling in the block
            x = MaxPooling1D(pool_size=pool_size, padding='same', name=f'resblock{block_num}_pool2')(x)
        
        # Shortcut connection: Adjust dimensions if necessary
        if shortcut.shape[-1] != filters or shortcut.shape[-2] != x.shape[-2]:
            shortcut = Conv1D(filters=filters, kernel_size=1, padding='same',
                              name=f'resblock{block_num}_shortcut_conv')(shortcut)
            shortcut = BatchNormalization(name=f'resblock{block_num}_shortcut_bn')(shortcut)
            
            # Adjust shortcut sequence length if pooling was applied in the main path
            if x_input.shape[-2] != x.shape[-2]:
                # This logic attempts to match pooling on shortcut to pooling on x
                # It assumes if x was pooled (once or twice), shortcut needs similar reduction.
                num_pools_in_x = 0
                if MaxPooling1D(pool_size=pool_size, padding='same')(x_input).shape[-2] == x.shape[-2] :
                    num_pools_in_x = 1
                elif MaxPooling1D(pool_size=pool_size, padding='same')(MaxPooling1D(pool_size=pool_size, padding='same')(x_input)).shape[-2] == x.shape[-2]:
                     num_pools_in_x = 2
                
                for p_idx in range(num_pools_in_x):
                    if shortcut.shape[-2] > x.shape[-2] : # only pool if shortcut is longer
                        shortcut = MaxPooling1D(pool_size=pool_size, padding='same', name=f'resblock{block_num}_shortcut_pool{p_idx+1}')(shortcut)


        x = Add(name=f'resblock{block_num}_add_shortcut')([shortcut, x])
        x = LeakyReLU(negative_slope=self.leaky_relu_alpha_after_residual_add,
                      name=f'resblock{block_num}_relu_after_add')(x)
        return x

    def build_model(self) -> Model:
        """
        Constructs and compiles the Keras model based on Version 5 architecture.
        Input, LSTM, and Output Dense layers are set to use 'float32' dtype.
        """
        # --- Input Layer ---
        input_layer = Input(
            shape=(self.time_steps, self.num_features),
            dtype='float32', # Explicitly set dtype to float32
            name='input_time_series'
        )
        x = input_layer

        # --- Convolutional Feature Extraction Backbone (Residual Blocks) ---
        for i, block_config_dict in enumerate(self.block_configs):
            filters = block_config_dict.get('filters', 64)
            kernel_size = block_config_dict.get('kernel_size', 3)
            pool_size = block_config_dict.get('pool_size', None)
            x = self._residual_block(x, filters, kernel_size, pool_size, block_num=i)

        # --- Bidirectional LSTM Block ---
        for i in range(self.num_bilstm_layers):
            x = Bidirectional(
                LSTM(
                    units=self.lstm_units,
                    return_sequences=True, # Necessary for subsequent Attention layer
                    recurrent_dropout=self.recurrent_dropout_lstm, # Will be 0.0 (off)
                    kernel_regularizer=self.lstm_kernel_regularizer,
                    dtype='float32' # Explicitly set dtype to float32 for LSTM
                ), name=f'bilstm_layer_{i+1}'
            )(x)
            if self.use_batchnorm_after_lstm:
                 x = BatchNormalization(name=f'bilstm_bn_{i+1}')(x)

        # --- Final MultiHeadAttention Layer (after LSTMs) ---
        # This was a MultiHeadAttention in the original V5 ModelBuilder
        final_mha_layer = MultiHeadAttention(
            num_heads=self.num_heads_final_mha,
            key_dim=self.key_dim_final_mha,
            name='final_mha_after_lstm'
        )
        x = final_mha_layer(query=x, value=x, key=x) # Self-attention
        if self.use_batchnorm_after_final_mha:
            x = BatchNormalization(name='final_mha_bn')(x)

        # --- Output Processing ---
        x = Flatten(name='flatten_mha_output')(x)

        # Optional Intermediate Dense Layer (was commented out in original V5)
        if self.use_intermediate_dense:
            x = Dense(
                units=self.intermediate_dense_units,
                kernel_regularizer=self.intermediate_dense_kernel_regularizer,
                name='intermediate_dense'
            )(x)
            if self.use_batchnorm_intermediate_dense:
                x = BatchNormalization(name='intermediate_dense_bn')(x)
            x = LeakyReLU(negative_slope=self.leaky_relu_alpha_intermediate_dense,
                          name='intermediate_dense_leaky_relu')(x)

        # Optional Dropout before the final dense layer
        if self.use_dropout_before_output: # Will be False by default
            x = Dropout(self.dropout_rate_before_output, name='dropout_before_final_dense')(x)

        # Final Dense layer for regression output
        output = Dense(
            units=1,
            activation=self.output_activation,
            kernel_regularizer=self.output_kernel_regularizer,
            name='output_dense',
            dtype='float32' # Explicitly set dtype to float32 for the output layer
        )(x)

        # --- Model Definition ---
        model = Model(inputs=input_layer, outputs=output, name='TimeSeriesForecastingModel_V5')

        # --- Optimizer Configuration ---
        optimizer_kwargs = {'learning_rate': self.optimizer_lr}
        if self.optimizer_weight_decay is not None:
            optimizer_kwargs['weight_decay'] = self.optimizer_weight_decay
        if self.optimizer_clipnorm is not None:
            optimizer_kwargs['clipnorm'] = self.optimizer_clipnorm
        if self.optimizer_clipvalue is not None:
            optimizer_kwargs['clipvalue'] = self.optimizer_clipvalue
        
        optimizer = AdamW(**optimizer_kwargs)

        # --- Model Compilation ---
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        return model

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the ModelBuilder."""
        config = {
            "time_steps": self.time_steps,
            "num_features": self.num_features,
            "block_configs": self.block_configs,
            "num_heads_residual_block": self.num_heads_residual_block,
            "key_dim_residual_block": self.key_dim_residual_block,
            "leaky_relu_alpha_conv_1": self.leaky_relu_alpha_conv_1,
            "leaky_relu_alpha_conv_2": self.leaky_relu_alpha_conv_2,
            "leaky_relu_alpha_after_residual_add": self.leaky_relu_alpha_after_residual_add,
            "conv_l2_reg": self.conv_kernel_regularizer.l2 if self.conv_kernel_regularizer else 0.0,
            "num_bilstm_layers": self.num_bilstm_layers,
            "lstm_units": self.lstm_units,
            "recurrent_dropout_lstm": self.recurrent_dropout_lstm,
            "lstm_l2_reg": self.lstm_kernel_regularizer.l2 if self.lstm_kernel_regularizer else 0.0,
            "use_batchnorm_after_lstm": self.use_batchnorm_after_lstm,
            "num_heads_final_mha": self.num_heads_final_mha,
            "key_dim_final_mha": self.key_dim_final_mha,
            "use_batchnorm_after_final_mha": self.use_batchnorm_after_final_mha,
            "use_intermediate_dense": self.use_intermediate_dense,
            "intermediate_dense_units": self.intermediate_dense_units,
            "leaky_relu_alpha_intermediate_dense": self.leaky_relu_alpha_intermediate_dense,
            "intermediate_dense_l2_reg": self.intermediate_dense_kernel_regularizer.l2 if self.intermediate_dense_kernel_regularizer else 0.0,
            "use_batchnorm_intermediate_dense": self.use_batchnorm_intermediate_dense,
            "use_dropout_before_output": self.use_dropout_before_output,
            "dropout_rate_before_output": self.dropout_rate_before_output,
            "output_activation": self.output_activation,
            "output_l2_reg": self.output_kernel_regularizer.l2 if self.output_kernel_regularizer else 0.0,
            "optimizer_lr": self.optimizer_lr,
            "optimizer_weight_decay": self.optimizer_weight_decay,
            "optimizer_clipnorm": self.optimizer_clipnorm,
            "optimizer_clipvalue": self.optimizer_clipvalue
        }
        return config
