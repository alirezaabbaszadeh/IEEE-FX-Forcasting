# Run.py for Version 1 (no csv/1/)
# Main script to configure, execute, and manage the time series forecasting pipeline.
# This script emphasizes clarity, reproducibility, and centralized hyperparameter management,
# aligning with best practices for research and drawing structural inspiration from "no csv/8/Run.py".

import os
import sys
import logging
import datetime # Not strictly used in this version but often useful for naming
import json # For potentially saving/loading configurations if extended
import argparse # For command-line argument parsing for enhanced flexibility

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras import backend as K

# Import the main orchestrator class
# Ensure that MainClass.py is in the same directory or accessible via PYTHONPATH
from MainClass import TimeSeriesModel
# ModelBuilder for Version 1 will be instantiated by MainClass based on parameters passed.

# ---------------------------------------------------------------------------
# Script Constants and Configuration
# ---------------------------------------------------------------------------
# --- Seeds for Reproducibility ---
# Setting seeds for all relevant libraries helps in achieving reproducible results,
# which is crucial for research.
SEED = 42 # A commonly used seed value; can be any integer.
tf.random.set_seed(SEED)
# If using numpy for other random operations, uncomment and set its seed too:
# import numpy as np
# np.random.seed(SEED)
# If using Python's built-in random module, uncomment and set its seed:
# import random
# random.seed(SEED)

# --- Logging Configuration ---
# Setup a logger for this script. More detailed logging (e.g., to a file per run)
# is typically handled within the TimeSeriesModel class.
SCRIPT_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment Setup Function
# ---------------------------------------------------------------------------
def setup_environment(enable_mixed_precision: bool = True,
                      mixed_precision_policy_name: str = 'mixed_float16'):
    """
    Configures the global TensorFlow environment, GPU settings, and basic application logging.

    Args:
        enable_mixed_precision (bool): If True, attempts to enable mixed precision.
                                       This can improve performance on compatible GPUs (e.g., NVIDIA Tensor Cores)
                                       but might require careful testing for numerical stability.
        mixed_precision_policy_name (str): The mixed precision policy to apply (e.g., 'mixed_float16').
    """
    # Configure base logging for the application (console output).
    # The format includes timestamp, log level, logger name, message, and source file/line.
    logging.basicConfig(
        level=logging.INFO, # Set to logging.DEBUG for more verbose output during development.
        format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s (%(filename)s:%(lineno)d)',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout) # Directs logs to the console.
        ]
    )
    SCRIPT_LOGGER.info(f"Global TensorFlow random seed set to {SEED} for potential reproducibility.")

    # Clear Keras session to ensure a clean state before starting a new model run.
    # This is important to prevent interference between different model runs in the same session.
    K.clear_session()
    SCRIPT_LOGGER.info("Keras session cleared.")

    # Reset TensorFlow's default graph. This is more relevant for TF1 compatibility mode
    # or when running multiple distinct model construction phases in the same script.
    tf.compat.v1.reset_default_graph()
    SCRIPT_LOGGER.info("TensorFlow default graph reset.")

    # GPU memory management: Enable memory growth for each detected GPU.
    # This prevents TensorFlow from allocating all GPU memory upfront, allowing for
    # more flexible GPU memory usage, especially when sharing GPUs.
    physical_gpus = tf.config.experimental.list_physical_devices('GPU')
    if physical_gpus:
        try:
            for gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            SCRIPT_LOGGER.info(f"GPU memory growth enabled for {len(physical_gpus)} GPU(s): {[gpu.name for gpu in physical_gpus]}")
        except RuntimeError as e:
            SCRIPT_LOGGER.error(f"Could not set GPU memory growth: {e}", exc_info=True)
    else:
        SCRIPT_LOGGER.info("No GPUs detected by TensorFlow. Model will run on CPU.")

    # Optional: Mixed Precision Policy for performance.
    if enable_mixed_precision:
        try:
            policy = mixed_precision.Policy(mixed_precision_policy_name)
            mixed_precision.set_global_policy(policy)
            SCRIPT_LOGGER.info(
                f"Mixed precision policy '{mixed_precision_policy_name}' successfully set. "
                f"Compute dtype: {policy.compute_dtype}, Variable dtype: {policy.variable_dtype}"
            )
        except Exception as e:
            SCRIPT_LOGGER.warning(
                f"Could not enable mixed precision policy '{mixed_precision_policy_name}'. Error: {e}. "
                "Training will use default precision (float32).", exc_info=True
            )

# ---------------------------------------------------------------------------
# Command-Line Argument Parsing
# ---------------------------------------------------------------------------
def parse_arguments():
    """
    Parses command-line arguments for the script.
    This allows for overriding default configurations for experiments without code changes.
    """
    parser = argparse.ArgumentParser(
        description="Run Time Series Forecasting Model (Version 1 - Standard Configuration)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    # Data and I/O arguments
    parser.add_argument('--data_file', type=str,
                        help="Path to the CSV data file.")
    parser.add_argument('--output_dir', type=str,
                        help="Base directory for saving run outputs (logs, models, plots).")
    parser.add_argument('--time_steps', type=int,
                        help="Number of time steps (look-back window) for input sequences.")

    # Training arguments
    parser.add_argument('--epochs', type=int, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, help="Batch size for training.")

    # Model architecture arguments (specific to ModelBuilder V1)
    # These should correspond to parameters in ModelBuilder V1's __init__
    parser.add_argument('--filters_conv1', type=int, help="Filters for Conv1D layer 1.")
    parser.add_argument('--kernel_size_conv1', type=int, help="Kernel size for Conv1D layer 1.")
    parser.add_argument('--leaky_alpha_conv1', type=float, help="Alpha for LeakyReLU in Conv1D block 1.")
    # ... (add similar arguments for filters_conv2, kernel_size_conv2, leaky_alpha_conv2)
    parser.add_argument('--lstm_units', type=int, help="Number of units in LSTM layers.")
    parser.add_argument('--recurrent_dropout_lstm', type=float, help="Recurrent dropout for LSTM.")
    parser.add_argument('--optimizer_lr', type=float, help="Learning rate for the AdamW optimizer.")
    parser.add_argument('--l2_reg_output', type=float, help="L2 regularization for the output layer.")

    # Boolean flags for optional features
    parser.add_argument('--enable_mixed_precision', action='store_true',
                        help="Enable mixed precision training (if GPU supports it).")

    args = parser.parse_args()
    return args

# ---------------------------------------------------------------------------
# Main Execution Function
# ---------------------------------------------------------------------------
def main(cli_args):
    """
    Defines configurations (from defaults and CLI overrides), initializes,
    and runs the Version 1 model pipeline.
    """
    SCRIPT_LOGGER.info(f"--- Initializing Model Pipeline for Version 1 (Standard Run) ---")

    # --- Base Directory of this script ---
    script_base_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Default Configuration Parameters ---
    # These serve as fallback values if not overridden by command-line arguments.

    # Data Loading Parameters
    default_data_file_path = os.path.join(script_base_dir, "csv/EURUSD_Candlestick_1_Hour_BID_01.01.2010-18.02.2025.csv")
    default_time_steps = 3       # Default from original DataLoader V1
    default_train_ratio = 0.94
    default_val_ratio = 0.03
    default_test_ratio = 0.03

    # Training Hyperparameters (Patterned after Version 8 for consistency in comparison)
    default_epochs = 60
    default_batch_size = 5000

    # Model Architecture Hyperparameters for Version 1 ModelBuilder
    # These keys MUST match the parameter names in the __init__ of the refactored ModelBuilder V1.
    default_model_builder_params_v1 = {
        'filters_conv1': 8,
        'kernel_size_conv1': 3,
        'leaky_relu_alpha_conv1': 0.04,     # Patterned after V8's leaky_relu_alpha_res_block
        'use_pooling_conv1': False,         # V1 original had pooling commented out
        'pool_size_conv1': None,
        'conv1_l2_reg': 0.0,

        'filters_conv2': 8,
        'kernel_size_conv2': 3,
        'leaky_relu_alpha_conv2': 0.03,     # Patterned after V8's leaky_relu_alpha_res_block2
        'use_pooling_conv2': False,         # V1 original had pooling commented out
        'pool_size_conv2': None,
        'conv2_l2_reg': 0.0,

        'num_bilstm_layers': 1,             # V1 original had one BiLSTM layer
        'lstm_units': 200,                  # Matches V8's lstm_units and V1's original
        'recurrent_dropout_lstm': False,      # V1 specific
        'lstm_l2_reg': False,
        'use_batchnorm_after_lstm': False,

        'use_batchnorm_after_attention': True, # V1 original had BN after Attention

        'use_dropout_before_output': False, # V1 original had dropout commented out
        'dropout_rate_before_output': 0.0,
        'output_activation': 'linear',
        'output_l2_reg': 0.0,               # Patterned after V8's l2_reg (effectively no L2)

        'optimizer_lr': 0.01,               # Patterned after V8's optimizer_lr
        'optimizer_weight_decay': None,     # Optional for AdamW
        'optimizer_clipnorm': None,         # Optional gradient clipping
        'optimizer_clipvalue': None
    }

    # --- Override defaults with CLI arguments if provided ---
    data_file_path = cli_args.data_file if cli_args.data_file else default_data_file_path
    time_steps = cli_args.time_steps if cli_args.time_steps is not None else default_time_steps
    epochs = cli_args.epochs if cli_args.epochs is not None else default_epochs
    batch_size = cli_args.batch_size if cli_args.batch_size is not None else default_batch_size

    # Update model_builder_params with CLI args for specific model hyperparameters
    model_builder_params_v1 = default_model_builder_params_v1.copy()
    if cli_args.optimizer_lr is not None:
        model_builder_params_v1['optimizer_lr'] = cli_args.optimizer_lr
    if cli_args.filters_conv1 is not None:
        model_builder_params_v1['filters_conv1'] = cli_args.filters_conv1
    if cli_args.kernel_size_conv1 is not None:
        model_builder_params_v1['kernel_size_conv1'] = cli_args.kernel_size_conv1
    if cli_args.leaky_alpha_conv1 is not None:
        model_builder_params_v1['leaky_relu_alpha_conv1'] = cli_args.leaky_alpha_conv1
    if cli_args.lstm_units is not None:
        model_builder_params_v1['lstm_units'] = cli_args.lstm_units
    if cli_args.recurrent_dropout_lstm is not None:
        model_builder_params_v1['recurrent_dropout_lstm'] = cli_args.recurrent_dropout_lstm
    if cli_args.l2_reg_output is not None:
        model_builder_params_v1['output_l2_reg'] = cli_args.l2_reg_output
    # Add more overrides here for other CLI arguments related to model_builder_params

    # `block_configs` is not used by ModelBuilder V1's specific design as outlined previously.
    # It's maintained as an empty list for potential compatibility with TimeSeriesModel's signature.
    block_configs_v1 = []

    # --- Output Directory for this Run ---
    # A descriptive name for the base directory where all experimental runs for V1 will be saved.
    # TimeSeriesModel will create a unique timestamped subdirectory within this.
    default_output_base_dir = os.path.join(script_base_dir, "IEEE_TNNLS_Runs_V1_Standard")
    output_base_dir = cli_args.output_dir if cli_args.output_dir else default_output_base_dir
    os.makedirs(output_base_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Log Final Configuration and Perform Sanity Checks
    # -----------------------------------------------------------------------
    SCRIPT_LOGGER.info(f"--- Final Configuration for Version 1 Run ---")
    SCRIPT_LOGGER.info(f"  Data file path: {data_file_path}")
    if not os.path.exists(data_file_path):
        SCRIPT_LOGGER.error(f"CRITICAL: Data file not found: {data_file_path}. Please verify the path. Exiting.")
        sys.exit(1) # Exit if the data file is crucial and missing.

    SCRIPT_LOGGER.info(f"  Run output base directory: {output_base_dir}")
    SCRIPT_LOGGER.info(f"  Training Parameters: Epochs={epochs}, Batch Size={batch_size}")
    SCRIPT_LOGGER.info(f"  Data Processing Parameters: Time Steps={time_steps}, Train Ratio={default_train_ratio}, Validation Ratio={default_val_ratio}, Test Ratio={default_test_ratio}")
    SCRIPT_LOGGER.info(f"  ModelBuilder V1 - Specific Architectural Parameters (passed via model_builder_params):")
    for key, value in model_builder_params_v1.items():
        SCRIPT_LOGGER.info(f"    {key}: {value}")
    # SCRIPT_LOGGER.info(f"  ModelBuilder V1 - Block Configurations (if used): {json.dumps(block_configs_v1, indent=4)}")


    # -----------------------------------------------------------------------
    # Initialize and Execute the Time Series Model Pipeline
    # -----------------------------------------------------------------------
    SCRIPT_LOGGER.info("Initializing TimeSeriesModel pipeline with the final configuration for Version 1...")
    try:
        # Instantiate the main pipeline orchestrator
        time_series_pipeline = TimeSeriesModel(
            # Data loading and splitting parameters
            file_path=data_file_path,
            time_steps=time_steps,
            train_ratio=default_train_ratio, # These could also be made CLI args
            val_ratio=default_val_ratio,
            test_ratio=default_test_ratio,

            # Output directory management
            base_dir=output_base_dir, # TimeSeriesModel will create a unique run-specific subdir here

            # Training hyperparameters
            epochs=epochs,
            batch_size=batch_size,

            # Model architecture parameters
            # For V1, architectural details are primarily passed via model_builder_params.
            # block_configs is included for signature consistency if MainClass expects it.
            block_configs=block_configs_v1,
            model_builder_params=model_builder_params_v1
        )

        SCRIPT_LOGGER.info("🚀 Starting the pipeline execution for Version 1...")
        # The .run() method encapsulates the entire workflow:
        # data loading -> model building -> training -> evaluation -> saving results.
        time_series_pipeline.run()
        SCRIPT_LOGGER.info("🎉 Pipeline execution for Version 1 completed successfully.")

    except Exception as e:
        SCRIPT_LOGGER.critical(
            f"❌ A critical error occurred during the pipeline execution for Version 1: {e}",
            exc_info=True # This will include the full stack trace in the log for debugging.
        )
        sys.exit(1) # Indicate failure to the calling environment (e.g., a batch script).

# ---------------------------------------------------------------------------
# Script Entry Point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # 1. Parse command-line arguments. This allows for dynamic configuration.
    args = parse_arguments()

    # 2. Setup the environment (logging, GPU, TensorFlow settings).
    # Mixed precision can be enabled via CLI if the argument is added to parse_arguments.
    setup_environment(enable_mixed_precision=args.enable_mixed_precision)

    # 3. Run the main pipeline function with parsed arguments.
    main(args)
