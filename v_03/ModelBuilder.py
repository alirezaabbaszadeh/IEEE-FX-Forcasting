# ModelBuilder.py for Version 3 (no csv/3/)
# Defines the neural network architecture for time series forecasting.
# This version features configurable residual blocks (defined by block_configs)
# with internal MultiHeadAttention, followed by BiLSTMs, a final Attention layer,
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
    Dropout, # Kept for potential future use, though default is off
    LeakyReLU,
    BatchNormalization,
    MultiHeadAttention,
    Add,
    Attention # Keras core Attention layer
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW
from typing import List, Dict, Any, Optional

# Optional: Set a global seed for TensorFlow operations if desired for this module.
# However, it's generally recommended to set seeds at the main script level (e.g., in Run.py)
# for overall experiment reproducibility.
SEED = 42
tf.random.set_seed(SEED)

class ModelBuilder:
    """
    Builds a time series forecasting model with Version 3 architecture:
    Input -> [Configurable Residual Blocks with Conv1D & MHA] -> BiLSTM_Block -> Attention -> Dense_Output.

    The convolutional part is defined by `block_configs`. Other components'
    hyperparameters are configurable via the constructor. Specific layers (Input, LSTM, Output Dense)
    are set to use 'float32' dtype. Dropout and recurrent_dropout default to off.
    """

    def __init__(self,
                 time_steps: int,
                 num_features: int,
                 block_configs: List[Dict[str, Any]],
                 # Parameters for components within each residual block
                 num_heads_residual_block: int = 12, # Renamed from num_heads_conv_block for clarity
                 key_dim_residual_block: int = 4,    # Renamed from key_dim_conv_block
                 leaky_relu_alpha_conv_1: float = 0.04,
                 leaky_relu_alpha_conv_2: float = 0.03,
                 leaky_relu_alpha_after_residual_add: float = 0.03, # New for V3's specific structure
                 conv_l2_reg: float = 0.0,
                 # BiLSTM Block Parameters
                 num_bilstm_layers: int = 1,
                 lstm_units: int = 200,
                 recurrent_dropout_lstm: float = 0.0, # Defaulting to 0.0 (off)
                 lstm_l2_reg: float = 0.0,
                 use_batchnorm_after_lstm: bool = True, # Added for flexibility
                 # Final Attention Layer (after LSTMs) Parameters
                 use_batchnorm_after_final_attention: bool = True, # As per V3 original structure
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
        Initializes the ModelBuilder for Version 3.

        Args:
            time_steps (int): Number of time steps in the input sequence.
            num_features (int): Number of features at each time step.
            block_configs (List[Dict[str, Any]]): Defines convolutional residual blocks.
                Expected keys per dict: 'filters', 'kernel_size', 'pool_size' (Optional).
            num_heads_residual_block (int): Num heads for MultiHeadAttention in residual blocks.
            key_dim_residual_block (int): Key dimension for MultiHeadAttention in residual blocks.
            leaky_relu_alpha_conv_1 (float): Alpha for the 1st LeakyReLU in a residual block's conv path.
            leaky_relu_alpha_conv_2 (float): Alpha for the 2nd LeakyReLU in a residual block's conv path.
            leaky_relu_alpha_after_residual_add (float): Alpha for LeakyReLU after the residual sum.
            conv_l2_reg (float): L2 regularization for Conv1D layers in residual blocks.
            num_bilstm_layers (int): Number of stacked Bidirectional LSTM layers.
            lstm_units (int): Number of units in each LSTM layer.
            recurrent_dropout_lstm (float): Recurrent dropout rate for LSTM layers (defaults to 0.0).
            lstm_l2_reg (float): L2 regularization for LSTM layers' kernels.
            use_batchnorm_after_lstm (bool): Whether to apply BatchNormalization after each LSTM layer.
            use_batchnorm_after_final_attention (bool): Apply BatchNormalization after the final Attention layer.
            use_dropout_before_output (bool): Apply Dropout before the final Dense output layer (defaults to False).
            dropout_rate_before_output (float): Dropout rate if use_dropout_before_output is True.
            output_activation (str): Activation function for the output Dense layer.
            output_l2_reg (float): L2 regularization for the output Dense layer's kernel.
            optimizer_lr (float): Learning rate for the AdamW optimizer.
            optimizer_weight_decay (Optional[float]): Weight decay for AdamW.
            optimizer_clipnorm (Optional[float]): Global norm for gradient clipping.
            optimizer_clipvalue (Optional[float]): Value for gradient clipping.
        """
        self.time_steps = time_steps
        self.num_features = num_features
        self.block_configs = block_configs

        # Residual block specific parameters
        self.num_heads_residual_block = num_heads_residual_block
        self.key_dim_residual_block = key_dim_residual_block
        self.leaky_relu_alpha_conv_1 = leaky_relu_alpha_conv_1
        self.leaky_relu_alpha_conv_2 = leaky_relu_alpha_conv_2
        self.leaky_relu_alpha_after_residual_add = leaky_relu_alpha_after_residual_add
        self.conv_kernel_regularizer = l2(conv_l2_reg) if conv_l2_reg > 0 else None

        # BiLSTM Block parameters
        self.num_bilstm_layers = num_bilstm_layers
        self.lstm_units = lstm_units
        self.recurrent_dropout_lstm = recurrent_dropout_lstm # Will be 0.0 by default
        self.lstm_kernel_regularizer = l2(lstm_l2_reg) if lstm_l2_reg > 0 else None
        self.use_batchnorm_after_lstm = use_batchnorm_after_lstm

        # Final Attention Layer parameters
        self.use_batchnorm_after_final_attention = use_batchnorm_after_final_attention

        # Flatten and Dense Output parameters
        self.use_dropout_before_output = use_dropout_before_output # Will be False by default
        self.dropout_rate_before_output = dropout_rate_before_output if self.use_dropout_before_output else 0.0
        self.output_activation = output_activation
        self.output_kernel_regularizer = l2(output_l2_reg) if output_l2_reg > 0 else None

        # Optimizer parameters
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
        Constructs a residual block as defined in Version 3's architecture.
        Each block contains two Conv1D-MHA sequences and a residual connection.

        Args:
            x_input (tf.Tensor): Input tensor to the residual block.
            filters (int): Number of filters for the Conv1D layers in this block.
            kernel_size (int): Kernel size for the Conv1D layers in this block.
            pool_size (Optional[int]): Pooling size for MaxPooling1D. If None, no pooling.
                                     The original V3 ModelBuilder applies pooling twice if specified.
            block_num (int): Identifier for the block, used for unique layer naming.

        Returns:
            tf.Tensor: Output tensor of the residual block.
        """
        shortcut = x_input

        # First Conv-MHA sequence
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same',
                   kernel_regularizer=self.conv_kernel_regularizer, name=f'resblock{block_num}_conv1')(x_input)
        x = BatchNormalization(name=f'resblock{block_num}_bn1')(x)
        x = LeakyReLU(negative_slope=self.leaky_relu_alpha_conv_1, name=f'resblock{block_num}_relu1')(x)

        mha1 = MultiHeadAttention(num_heads=self.num_heads_residual_block, key_dim=self.key_dim_residual_block,
                                  name=f'resblock{block_num}_mha1')
        x = mha1(query=x, value=x, key=x)
        x = BatchNormalization(name=f'resblock{block_num}_bn_after_mha1')(x)

        # if pool_size is not None and pool_size > 1:
        #     x = MaxPooling1D(pool_size=pool_size, padding='same', name=f'resblock{block_num}_pool1')(x)

        # Second Conv-MHA sequence
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same',
                   kernel_regularizer=self.conv_kernel_regularizer, name=f'resblock{block_num}_conv2')(x)
        x = BatchNormalization(name=f'resblock{block_num}_bn2')(x)
        x = LeakyReLU(negative_slope=self.leaky_relu_alpha_conv_2, name=f'resblock{block_num}_relu2')(x)

        mha2 = MultiHeadAttention(num_heads=self.num_heads_residual_block, key_dim=self.key_dim_residual_block,
                                  name=f'resblock{block_num}_mha2')
        x = mha2(query=x, value=x, key=x)
        x = BatchNormalization(name=f'resblock{block_num}_bn_after_mha2')(x)

        # if pool_size is not None and pool_size > 1: # Second pooling if specified
        #     x = MaxPooling1D(pool_size=pool_size, padding='same', name=f'resblock{block_num}_pool2')(x)

        # Shortcut connection: Adjust dimensions if necessary
        # This is important if pooling changed the sequence length or if filter counts differ.
        if shortcut.shape[-2] != x.shape[-2] or shortcut.shape[-1] != x.shape[-1]: # Check both time and feature dims
            shortcut = Conv1D(filters=filters, kernel_size=1, padding='same',
                              name=f'resblock{block_num}_shortcut_conv')(shortcut)
            shortcut = BatchNormalization(name=f'resblock{block_num}_shortcut_bn')(shortcut)
            # If pooling was applied to x, shortcut also needs to be pooled to match time dimension
            if x_input.shape[-2] != x.shape[-2] and pool_size is not None and pool_size > 1 : # if x was pooled
                 # Apply pooling twice to shortcut if x was pooled twice
                 if MaxPooling1D(pool_size=pool_size, padding='same')(MaxPooling1D(pool_size=pool_size, padding='same')(x_input)).shape[-2] == x.shape[-2]:
                    shortcut = MaxPooling1D(pool_size=pool_size, padding='same', name=f'resblock{block_num}_shortcut_pool1')(shortcut)
                    shortcut = MaxPooling1D(pool_size=pool_size, padding='same', name=f'resblock{block_num}_shortcut_pool2')(shortcut)
                 # Apply pooling once if x was pooled once
                 elif MaxPooling1D(pool_size=pool_size, padding='same')(x_input).shape[-2] == x.shape[-2]:
                    shortcut = MaxPooling1D(pool_size=pool_size, padding='same', name=f'resblock{block_num}_shortcut_pool1')(shortcut)


        x = Add(name=f'resblock{block_num}_add_shortcut')([shortcut, x])
        x = LeakyReLU(negative_slope=self.leaky_relu_alpha_after_residual_add,
                      name=f'resblock{block_num}_relu_after_add')(x)
        return x

    def build_model(self) -> Model:
        """
        Constructs and compiles the Keras model based on Version 3 architecture.
        Input, LSTM, and Output Dense layers are set to use 'float32' dtype.

        Returns:
            tf.keras.models.Model: The compiled Keras model.
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
            filters = block_config_dict.get('filters', 8) # Default filters if not specified
            kernel_size = block_config_dict.get('kernel_size', 3) # Default kernel_size
            pool_size = block_config_dict.get('pool_size', None) # Default no pooling

            x = self._residual_block(x, filters, kernel_size, pool_size, block_num=i)

        # --- Bidirectional LSTM Block ---
        for i in range(self.num_bilstm_layers):
            x = Bidirectional(
                LSTM(
                    units=self.lstm_units,
                    return_sequences=True, # True for all but last if stacking, or if Attention follows
                    recurrent_dropout=self.recurrent_dropout_lstm, # Will be 0.0
                    kernel_regularizer=self.lstm_kernel_regularizer,
                    dtype='float32' # Explicitly set dtype to float32 for LSTM computations
                ), name=f'bilstm_layer_{i+1}'
            )(x)
            if self.use_batchnorm_after_lstm: # Apply BN after each BiLSTM layer
                 x = BatchNormalization(name=f'bilstm_bn_{i+1}')(x)
            # No dropout after LSTM as per new requirements (recurrent_dropout is 0.0)

        # --- Final Attention Layer (Keras Core Attention) ---
        attention_output = Attention(name='final_attention_over_lstm')([x, x])
        if self.use_batchnorm_after_final_attention:
            attention_output = BatchNormalization(name='final_attention_bn')(attention_output)
        x = attention_output

        # --- Output Processing ---
        x = Flatten(name='flatten_attention_output')(x)

        if self.use_dropout_before_output: # This will be False by default
            x = Dropout(self.dropout_rate_before_output, name='dropout_before_final_dense')(x)

        output = Dense(
            units=1,
            activation=self.output_activation,
            kernel_regularizer=self.output_kernel_regularizer,
            name='output_dense',
            dtype='float32' # Explicitly set dtype to float32 for the output layer
        )(x)

        # --- Model Definition ---
        model = Model(inputs=input_layer, outputs=output, name='TimeSeriesForecastingModel_V3')

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
            "use_batchnorm_after_final_attention": self.use_batchnorm_after_final_attention,
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
