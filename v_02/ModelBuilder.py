# ModelBuilder.py for Version 2 (no csv/2/)
# Defines the neural network architecture for time series forecasting.
# This version uses a list of block configurations for its convolutional part
# and allows other hyperparameters to be set externally.

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
    MultiHeadAttention, # Used within convolutional blocks
    Attention         # Keras core Attention layer, used after LSTMs
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
    Builds a time series forecasting model with Version 2 architecture:
    Input -> [Configurable Conv_Blocks with MHA] -> BiLSTM_Block -> Attention -> Dense_Output.

    The convolutional part is defined by `block_configs`. Other components'
    hyperparameters are configurable via the constructor.
    """

    def __init__(self,
                 time_steps: int,
                 num_features: int,
                 block_configs: List[Dict[str, Any]],
                 # Parameters for MultiHeadAttention within each convolutional block
                 num_heads_conv_block: int = 12,
                 key_dim_conv_block: int = 4,
                 # Parameters for LeakyReLU within each convolutional block
                 leaky_relu_alpha_conv_1: float = 0.04, # For the first LeakyReLU in a conv block pair
                 leaky_relu_alpha_conv_2: float = 0.03, # For the second LeakyReLU in a conv block pair
                 conv_l2_reg: float = 0.0, # Optional L2 for Conv1D layers within blocks
                 # BiLSTM Block Parameters
                 num_bilstm_layers: int = 1,
                 lstm_units: int = 200,
                 recurrent_dropout_lstm: Optional[float] = None,
                 lstm_l2_reg: float = 0.0, # Optional L2 for LSTM layers' kernels
                 # Final Attention Layer (after LSTMs) Parameters
                 use_batchnorm_after_final_attention: bool = True,
                 # Flatten and Dense Output Parameters
                 use_dropout_before_output: bool = False, # Original V2 did not have this explicitly
                 dropout_rate_before_output: float = 0.1,
                 output_activation: str = 'linear',
                 output_l2_reg: float = 0.0,
                 # Optimizer Parameters
                 optimizer_lr: float = 0.01,
                 optimizer_weight_decay: Optional[float] = None,
                 optimizer_clipnorm: Optional[float] = None,
                 optimizer_clipvalue: Optional[float] = None
                 ):
        """
        Initializes the ModelBuilder for Version 2.

        Args:
            time_steps (int): Number of time steps in the input sequence.
            num_features (int): Number of features at each time step.
            block_configs (List[Dict[str, Any]]): A list of dictionaries, where each
                dictionary configures a convolutional block. Expected keys per dict:
                'filters' (int), 'kernel_size' (int), 'pool_size' (Optional[int]).
            num_heads_conv_block (int): Num heads for MultiHeadAttention within conv blocks.
            key_dim_conv_block (int): Key dimension for MultiHeadAttention within conv blocks.
            leaky_relu_alpha_conv_1 (float): Alpha for the first LeakyReLU in a conv block.
            leaky_relu_alpha_conv_2 (float): Alpha for the second LeakyReLU in a conv block.
            conv_l2_reg (float): L2 regularization for Conv1D layers in blocks.
            num_bilstm_layers (int): Number of stacked Bidirectional LSTM layers.
            lstm_units (int): Number of units in each LSTM layer.
            recurrent_dropout_lstm (float): Recurrent dropout rate for LSTM layers.
            lstm_l2_reg (float): L2 regularization for LSTM layers' kernels.
            use_batchnorm_after_final_attention (bool): Apply BatchNormalization after the final Attention layer.
            use_dropout_before_output (bool): Apply Dropout before the final Dense output layer.
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

        # Convolutional block specific parameters (applied to each block)
        self.num_heads_conv_block = num_heads_conv_block
        self.key_dim_conv_block = key_dim_conv_block
        self.leaky_relu_alpha_conv_1 = leaky_relu_alpha_conv_1
        self.leaky_relu_alpha_conv_2 = leaky_relu_alpha_conv_2
        self.conv_kernel_regularizer = l2(conv_l2_reg) if conv_l2_reg > 0 else None

        # BiLSTM Block parameters
        self.num_bilstm_layers = num_bilstm_layers
        self.lstm_units = lstm_units
        self.recurrent_dropout_lstm = recurrent_dropout_lstm
        self.lstm_kernel_regularizer = l2(lstm_l2_reg) if lstm_l2_reg > 0 else None

        # Final Attention Layer parameters
        self.use_batchnorm_after_final_attention = use_batchnorm_after_final_attention

        # Flatten and Dense Output parameters
        self.use_dropout_before_output = use_dropout_before_output
        self.dropout_rate_before_output = dropout_rate_before_output
        self.output_activation = output_activation
        self.output_kernel_regularizer = l2(output_l2_reg) if output_l2_reg > 0 else None

        # Optimizer parameters
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.optimizer_clipnorm = optimizer_clipnorm
        self.optimizer_clipvalue = optimizer_clipvalue

    def build_model(self) -> Model:
        """
        Constructs and compiles the Keras model based on Version 2 architecture
        and initialized hyperparameters.

        Returns:
            tf.keras.models.Model: The compiled Keras model.
        """
        # --- Input Layer ---
        input_layer = Input(
            shape=(self.time_steps, self.num_features),
            name='input_time_series',
            dtype='float32'
        )
        x = input_layer

        # --- Convolutional Blocks ---
        # Iterates through the configurations provided in `block_configs`.
        # Each block consists of two Conv1D-MHA sequences and optional pooling.
        for i, block_conf in enumerate(self.block_configs):
            filters = block_conf.get('filters', 32) # Default if not in config
            kernel_size = block_conf.get('kernel_size', 3) # Default if not in config
            pool_size = block_conf.get('pool_size', None) # Default if not in config

            # First Conv1D -> BN -> LeakyReLU -> MHA -> BN sequence in the block
            x = Conv1D(
                filters=filters, kernel_size=kernel_size, padding='same',
                kernel_regularizer=self.conv_kernel_regularizer, name=f'block{i}_conv1a')(x)
            x = BatchNormalization(name=f'block{i}_bn1a')(x)
            x = LeakyReLU(alpha=self.leaky_relu_alpha_conv_1, name=f'block{i}_relu1a')(x)
            
            mha_layer_1a = MultiHeadAttention(
                num_heads=self.num_heads_conv_block, key_dim=self.key_dim_conv_block,
                name=f'block{i}_mha1a')
            x = mha_layer_1a(query=x, value=x, key=x)
            x = BatchNormalization(name=f'block{i}_bn_after_mha1a')(x)
            # Optional Dropout could be added here:
            # x = Dropout(self.dropout_rate)(x) # Assuming a general dropout_rate attribute

            # if pool_size is not None: # First pooling, if specified for the block
            #     x = MaxPooling1D(pool_size=pool_size, padding='same', name=f'block{i}_pool1')(x)

            # Second Conv1D -> BN -> LeakyReLU -> MHA -> BN sequence in the block
            x = Conv1D(
                filters=filters, kernel_size=kernel_size, padding='same',
                kernel_regularizer=self.conv_kernel_regularizer, name=f'block{i}_conv1b')(x)
            x = BatchNormalization(name=f'block{i}_bn1b')(x)
            x = LeakyReLU(alpha=self.leaky_relu_alpha_conv_2, name=f'block{i}_relu1b')(x)

            mha_layer_1b = MultiHeadAttention(
                num_heads=self.num_heads_conv_block, key_dim=self.key_dim_conv_block,
                name=f'block{i}_mha1b')
            x = mha_layer_1b(query=x, value=x, key=x)
            x = BatchNormalization(name=f'block{i}_bn_after_mha1b')(x)
            # Optional Dropout could be added here

            # if pool_size is not None: # Second pooling, if specified for the block
            #     x = MaxPooling1D(pool_size=pool_size, padding='same', name=f'block{i}_pool2')(x)

        # --- Bidirectional LSTM Block ---
        for i in range(self.num_bilstm_layers):
            x = Bidirectional(
                LSTM(
                    units=self.lstm_units,
                    return_sequences=True, # Necessary for subsequent Attention layer
                    recurrent_dropout=self.recurrent_dropout_lstm,
                    kernel_regularizer=self.lstm_kernel_regularizer,
                    dtype='float32', 
                ), name=f'bilstm_layer_{i+1}'
            )(x)
            # Optional: BatchNormalization between LSTM layers if num_bilstm_layers > 1
            # if self.num_bilstm_layers > 1 and i < (self.num_bilstm_layers - 1):
            #     x = BatchNormalization(name=f'bilstm_inter_bn_{i+1}')(x)

        # --- Final Attention Layer (Keras Core Attention) ---
        # This layer computes a weighted sum of the LSTM outputs.
        attention_output = Attention(name='final_attention_over_lstm')([x, x]) # Self-attention on LSTM sequence
        if self.use_batchnorm_after_final_attention:
            attention_output = BatchNormalization(name='final_attention_bn')(attention_output)
        x = attention_output

        # --- Output Processing ---
        x = Flatten(name='flatten_attention_output')(x)

        if self.use_dropout_before_output:
            x = Dropout(self.dropout_rate_before_output, name='dropout_before_final_dense')(x)

        output = Dense(
            units=1, # Single output for regression
            activation=self.output_activation,
            kernel_regularizer=self.output_kernel_regularizer,
            name='output_dense',
            dtype='float32'
        )(x)

        # --- Model Definition ---
        model = Model(inputs=input_layer, outputs=output, name='TimeSeriesForecastingModel_V2')

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
            loss='mse', # Mean Squared Error for regression
            metrics=['mae', 'mse'] # Mean Absolute Error and MSE for monitoring
        )

        return model

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the ModelBuilder.
        Useful for saving and recreating the builder or for logging purposes.
        """
        return {
            "time_steps": self.time_steps,
            "num_features": self.num_features,
            "block_configs": self.block_configs,
            "num_heads_conv_block": self.num_heads_conv_block,
            "key_dim_conv_block": self.key_dim_conv_block,
            "leaky_relu_alpha_conv_1": self.leaky_relu_alpha_conv_1,
            "leaky_relu_alpha_conv_2": self.leaky_relu_alpha_conv_2,
            "conv_l2_reg": self.conv_kernel_regularizer.l2 if self.conv_kernel_regularizer else 0.0,
            "num_bilstm_layers": self.num_bilstm_layers,
            "lstm_units": self.lstm_units,
            "recurrent_dropout_lstm": self.recurrent_dropout_lstm,
            "lstm_l2_reg": self.lstm_kernel_regularizer.l2 if self.lstm_kernel_regularizer else 0.0,
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

# Example of how this ModelBuilder might be used (typically in MainClass.py):
# if __name__ == '__main__':
#     # Define sample parameters (these would come from Run.py)
#     sample_time_steps = 60
#     sample_num_features = 5
#     sample_block_configs_v2 = [
#         {'filters': 8, 'kernel_size': 3, 'pool_size': 2},
#         # {'filters': 16, 'kernel_size': 3, 'pool_size': None} # Another block example
#     ]
#     sample_model_builder_params_v2 = {
#         'num_heads_conv_block': 8, 'key_dim_conv_block': 4,
#         'leaky_relu_alpha_conv_1': 0.05, 'leaky_relu_alpha_conv_2': 0.02,
#         'num_bilstm_layers': 1, 'lstm_units': 150, 'recurrent_dropout_lstm': 0.25,
#         'output_l2_reg': 0.0001, 'optimizer_lr': 0.005
#     }
#
#     builder_v2 = ModelBuilder(
#         time_steps=sample_time_steps,
#         num_features=sample_num_features,
#         block_configs=sample_block_configs_v2,
#         **sample_model_builder_params_v2 # Unpack the dictionary of other params
#     )
#
#     model_v2 = builder_v2.build_model()
#     model_v2.summary(line_length=150)
#
#     print("\nModelBuilder V2 Configuration:")
#     print(json.dumps(builder_v2.get_config(), indent=2))
