# ModelBuilder.py for Version 1 (no csv/1/)
# Defines the neural network architecture for time series forecasting.
# This version is designed to be highly configurable through its constructor,
# allowing hyperparameters to be set externally (e.g., from Run.py).

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
    Attention # Using the Keras core Attention layer
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW # Using AdamW as in the original
from typing import Optional, Union, List, Dict, Any

# Optional: Set a global seed for TensorFlow operations in this module if desired,
# though it's generally better to set seeds at the main script level (e.g., in Run.py)
SEED = 42
tf.random.set_seed(SEED)

class ModelBuilder:
    """
    Builds a time series forecasting model with a specific architecture:
    Input -> Conv_Block1 -> Conv_Block2 -> BiLSTM_Block -> Attention -> Dense_Output.

    Each component's hyperparameters are configurable via the constructor.
    The architecture preserves the layer sequence from the original "no csv/1/ModelBuilder.py"
    while making its parameters externally controllable and aligned with best practices.
    """

    def __init__(self,
                 time_steps: int,
                 num_features: int,
                 # Convolutional Block 1 Parameters
                 filters_conv1: int = 8,
                 kernel_size_conv1: int = 3,
                 leaky_relu_alpha_conv1: float = 0.04,
                 use_pooling_conv1: bool = False, # Original had MaxPooling commented out
                 pool_size_conv1: int = 2,
                 conv1_l2_reg: float = 0.0, # Optional L2 for Conv1
                 # Convolutional Block 2 Parameters
                 filters_conv2: int = 8,
                 kernel_size_conv2: int = 3,
                 leaky_relu_alpha_conv2: float = 0.03,
                 use_pooling_conv2: bool = False, # Original had MaxPooling commented out
                 pool_size_conv2: int = 2,
                 conv2_l2_reg: float = 0.0, # Optional L2 for Conv2
                 # BiLSTM Block Parameters
                 num_bilstm_layers: int = 1, # Allowing for stacking BiLSTM layers
                 lstm_units: int = 200,
                 recurrent_dropout_lstm: float = 0.0,
                 lstm_l2_reg: float = 0.0, # Optional L2 for LSTM layers
                 use_batchnorm_after_lstm: bool = True, # Defaulting to True as in many architectures
                 # Attention Layer Parameters
                 use_batchnorm_after_attention: bool = True, # As per original V1 (BN after Attention)
                 # Flatten and Dense Output Parameters
                 use_dropout_before_output: bool = False, # Original had Dropout commented out
                 dropout_rate_before_output: float = 0.1,
                 output_activation: str = 'linear',
                 output_l2_reg: float = 0.0,
                 # Optimizer Parameters
                 optimizer_lr: float = 0.01, # Defaulting to V8's pattern
                 optimizer_weight_decay: Optional[float] = None, # For AdamW if specified
                 optimizer_clipnorm: Optional[float] = None,
                 optimizer_clipvalue: Optional[float] = None
                 # Removed dtype specifications from __init__; Keras handles this based on global policy or layer defaults.
                 ):
        """
        Initializes the ModelBuilder with architectural and optimizer hyperparameters.

        Args:
            time_steps (int): Number of time steps in the input sequence.
            num_features (int): Number of features per time step.

            filters_conv1 (int): Number of filters for the first Conv1D layer.
            kernel_size_conv1 (int): Kernel size for the first Conv1D layer.
            leaky_relu_alpha_conv1 (float): Alpha for LeakyReLU in the first Conv1D block.
            use_pooling_conv1 (bool): Whether to use MaxPooling1D after the first Conv1D block.
            pool_size_conv1 (int): Pool size if use_pooling_conv1 is True.
            conv1_l2_reg (float): L2 regularization factor for the first Conv1D layer's kernel.

            filters_conv2 (int): Number of filters for the second Conv1D layer.
            kernel_size_conv2 (int): Kernel size for the second Conv1D layer.
            leaky_relu_alpha_conv2 (float): Alpha for LeakyReLU in the second Conv1D block.
            use_pooling_conv2 (bool): Whether to use MaxPooling1D after the second Conv1D block.
            pool_size_conv2 (int): Pool size if use_pooling_conv2 is True.
            conv2_l2_reg (float): L2 regularization factor for the second Conv1D layer's kernel.

            num_bilstm_layers (int): Number of stacked Bidirectional LSTM layers.
            lstm_units (int): Number of units in each LSTM layer.
            recurrent_dropout_lstm (float): Recurrent dropout rate for LSTM layers.
            lstm_l2_reg (float): L2 regularization factor for LSTM layers' kernels.
            use_batchnorm_after_lstm (bool): Whether to apply BatchNormalization after LSTM block.

            use_batchnorm_after_attention (bool): Whether to apply BatchNormalization after Attention.

            use_dropout_before_output (bool): Whether to use a Dropout layer before the final Dense output.
            dropout_rate_before_output (float): Dropout rate if use_dropout_before_output is True.
            output_activation (str): Activation function for the output Dense layer.
            output_l2_reg (float): L2 regularization factor for the output Dense layer's kernel.

            optimizer_lr (float): Learning rate for the AdamW optimizer.
            optimizer_weight_decay (Optional[float]): Weight decay for AdamW. If None, default AdamW behavior.
            optimizer_clipnorm (Optional[float]): Global norm for gradient clipping.
            optimizer_clipvalue (Optional[float]): Value for gradient clipping.
        """
        self.time_steps = time_steps
        self.num_features = num_features

        # Convolutional Block 1 parameters
        self.filters_conv1 = filters_conv1
        self.kernel_size_conv1 = kernel_size_conv1
        self.leaky_relu_alpha_conv1 = leaky_relu_alpha_conv1
        self.use_pooling_conv1 = use_pooling_conv1
        self.pool_size_conv1 = pool_size_conv1
        self.conv1_kernel_regularizer = l2(conv1_l2_reg) if conv1_l2_reg > 0 else None

        # Convolutional Block 2 parameters
        self.filters_conv2 = filters_conv2
        self.kernel_size_conv2 = kernel_size_conv2
        self.leaky_relu_alpha_conv2 = leaky_relu_alpha_conv2
        self.use_pooling_conv2 = use_pooling_conv2
        self.pool_size_conv2 = pool_size_conv2
        self.conv2_kernel_regularizer = l2(conv2_l2_reg) if conv2_l2_reg > 0 else None

        # BiLSTM Block parameters
        self.num_bilstm_layers = num_bilstm_layers
        self.lstm_units = lstm_units
        self.recurrent_dropout_lstm = recurrent_dropout_lstm
        self.lstm_kernel_regularizer = l2(lstm_l2_reg) if lstm_l2_reg > 0 else None
        self.use_batchnorm_after_lstm = use_batchnorm_after_lstm

        # Attention Layer parameters
        self.use_batchnorm_after_attention = use_batchnorm_after_attention

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

        # Keras layers automatically handle dtype based on global policy (e.g., mixed_float16)
        # or default to float32. Explicit dtype='float32' can be set on layers if overriding policy is needed.

    def build_model(self) -> Model:
        """
        Constructs and compiles the Keras model based on the initialized hyperparameters.

        Returns:
            tf.keras.models.Model: The compiled Keras model.
        """
        # --- Input Layer ---
        # Defines the expected shape of the input data.
        input_layer = Input(
            shape=(self.time_steps, self.num_features),
            name='input_time_series',
            dtype='float32'
        )
        x = input_layer

        # --- Convolutional Block 1 ---
        # Extracts local patterns from the input sequence.
        x = Conv1D(
            filters=self.filters_conv1,
            kernel_size=self.kernel_size_conv1,
            padding='same', # 'same' padding ensures output length matches input length (before pooling)
            kernel_regularizer=self.conv1_kernel_regularizer,
            name='conv1d_block1_conv'
        )(x)
        x = BatchNormalization(name='conv1d_block1_bn')(x)
        x = LeakyReLU(alpha=self.leaky_relu_alpha_conv1, name='conv1d_block1_activation')(x)
        if self.use_pooling_conv1:
            x = MaxPooling1D(pool_size=self.pool_size_conv1, padding='same', name='conv1d_block1_pool')(x)

        # --- Convolutional Block 2 ---
        # Further feature extraction and pattern recognition.
        x = Conv1D(
            filters=self.filters_conv2,
            kernel_size=self.kernel_size_conv2,
            padding='same',
            kernel_regularizer=self.conv2_kernel_regularizer,
            name='conv1d_block2_conv'
        )(x)
        x = BatchNormalization(name='conv1d_block2_bn')(x)
        x = LeakyReLU(alpha=self.leaky_relu_alpha_conv2, name='conv1d_block2_activation')(x)
        if self.use_pooling_conv2:
            x = MaxPooling1D(pool_size=self.pool_size_conv2, padding='same', name='conv1d_block2_pool')(x)

        # --- Bidirectional LSTM Block ---
        # Captures temporal dependencies in both forward and backward directions.
        # Loops to create specified number of BiLSTM layers.
        for i in range(self.num_bilstm_layers):
            x = Bidirectional(
                LSTM(
                    units=self.lstm_units,
                    return_sequences=True, # Crucial for stacking LSTMs or feeding to Attention
                    recurrent_dropout=self.recurrent_dropout_lstm,
                    kernel_regularizer=self.lstm_kernel_regularizer,
                    dtype='float32'
                    # Keras handles dtype based on policy, explicit 'float32' can be used if needed
                ), name=f'bilstm_layer_{i+1}'
            )(x)
            if self.use_batchnorm_after_lstm and (i < self.num_bilstm_layers -1): # BN between LSTMs, not after last if attention follows
                 x = BatchNormalization(name=f'bilstm_layer_{i+1}_bn_inter')(x)


        # --- Attention Layer ---
        # Allows the model to weigh the importance of different time steps from LSTM output.
        # Using Keras core tf.keras.layers.Attention which computes a weighted sum.
        # For self-attention on LSTM outputs:
        attention_output = Attention(name='attention_over_lstm_outputs')([x, x]) # Query and Value are the same for self-attention context
        if self.use_batchnorm_after_attention:
            attention_output = BatchNormalization(name='attention_bn')(attention_output)
        x = attention_output

        # --- Output Processing ---
        # Flatten the sequence output from Attention to feed into Dense layers.
        x = Flatten(name='flatten_attention_output')(x)

        # Optional Dropout before the final dense layer for regularization.
        if self.use_dropout_before_output:
            x = Dropout(self.dropout_rate_before_output, name='dropout_before_output_dense')(x)

        # Final Dense layer for regression output.
        output = Dense(
            units=1, # Single output unit for forecasting one value
            activation=self.output_activation, # Typically 'linear' for regression
            kernel_regularizer=self.output_kernel_regularizer,
            name='output_dense',
            dtype='float32'
            # Keras handles dtype
        )(x)

        # --- Model Definition ---
        model = Model(inputs=input_layer, outputs=output, name='TimeSeriesForecastingModel_V1')

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
        # Configures the model for training.
        model.compile(
            optimizer=optimizer,
            loss='mse', # Mean Squared Error is a common loss for regression.
            metrics=['mae', 'mse'] # Mean Absolute Error and MSE for monitoring.
        )

        return model

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the ModelBuilder.
        Useful for saving and recreating the builder if needed, or for logging.
        """
        return {
            "time_steps": self.time_steps,
            "num_features": self.num_features,
            "filters_conv1": self.filters_conv1,
            "kernel_size_conv1": self.kernel_size_conv1,
            "leaky_relu_alpha_conv1": self.leaky_relu_alpha_conv1,
            "use_pooling_conv1": self.use_pooling_conv1,
            "pool_size_conv1": self.pool_size_conv1,
            "conv1_l2_reg": self.conv1_l2_reg.l2 if self.conv1_l2_reg else 0.0,
            "filters_conv2": self.filters_conv2,
            "kernel_size_conv2": self.kernel_size_conv2,
            "leaky_relu_alpha_conv2": self.leaky_relu_alpha_conv2,
            "use_pooling_conv2": self.use_pooling_conv2,
            "pool_size_conv2": self.pool_size_conv2,
            "conv2_l2_reg": self.conv2_l2_reg.l2 if self.conv2_l2_reg else 0.0,
            "num_bilstm_layers": self.num_bilstm_layers,
            "lstm_units": self.lstm_units,
            "recurrent_dropout_lstm": self.recurrent_dropout_lstm,
            "lstm_l2_reg": self.lstm_kernel_regularizer.l2 if self.lstm_kernel_regularizer else 0.0,
            "use_batchnorm_after_lstm": self.use_batchnorm_after_lstm,
            "use_batchnorm_after_attention": self.use_batchnorm_after_attention,
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
#     # Define some sample parameters (these would come from Run.py)
#     sample_time_steps = 60
#     sample_num_features = 5
#     sample_params_from_run_py = {
#         'filters_conv1': 16, 'kernel_size_conv1': 5, 'leaky_relu_alpha_conv1': 0.02,
#         'filters_conv2': 32, 'kernel_size_conv2': 5, 'leaky_relu_alpha_conv2': 0.02,
#         'lstm_units': 100, 'recurrent_dropout_lstm': 0.2, 'num_bilstm_layers': 1,
#         'output_l2_reg': 0.001,
#         'optimizer_lr': 0.001
#     }
#
#     builder = ModelBuilder(
#         time_steps=sample_time_steps,
#         num_features=sample_num_features,
#         **sample_params_from_run_py # Unpack the dictionary of params
#     )
#
#     model = builder.build_model()
#     model.summary(line_length=120)
#
#     print("\nModelBuilder Configuration:")
#     print(json.dumps(builder.get_config(), indent=2))