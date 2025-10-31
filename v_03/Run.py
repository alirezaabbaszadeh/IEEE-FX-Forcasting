# Run.py for Version 3 (no csv/3/)
# Main script to configure, execute, and manage the time series forecasting pipeline
# for Model Version 3. This script centralizes hyperparameter settings, drawing
# structural inspiration from Version 8's Run.py for consistency and best practices.
# Key features for V3: configurable residual blocks and specific dtype handling.

import os
import sys
import logging
import datetime # Useful for naming or logging
import json     # For pretty-printing dictionaries in logs
import argparse # For command-line argument parsing

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras import backend as K

# Import the main orchestrator class
# Ensure MainClass.py is in the same directory or accessible via PYTHONPATH
from MainClass import TimeSeriesModel
# ModelBuilder for Version 3 will be instantiated by MainClass based on parameters passed.

# ---------------------------------------------------------------------------
# Script Constants and Configuration
# ---------------------------------------------------------------------------
SEED = 42 # Seed for reproducibility
tf.random.set_seed(SEED)
# import numpy as np; np.random.seed(SEED) # If NumPy's random functions are used directly
# import random; random.seed(SEED) # If Python's random module is used directly

SCRIPT_LOGGER = logging.getLogger(__name__) # Logger specific to this Run.py script

# ---------------------------------------------------------------------------
# Environment Setup Function
# ---------------------------------------------------------------------------
def setup_environment(enable_mixed_precision: bool = True, # Enabled by default as per request
                      mixed_precision_policy_name: str = 'mixed_float16'):
    """
    Configures the global TensorFlow environment, GPU settings, and basic application logging.

    Args:
        enable_mixed_precision (bool): If True, attempts to enable mixed precision. Default is True.
        mixed_precision_policy_name (str): The mixed precision policy to apply.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s (%(filename)s:%(lineno)d)',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    SCRIPT_LOGGER.info(f"Global TensorFlow random seed set to {SEED}.")

    K.clear_session()
    SCRIPT_LOGGER.info("Keras session cleared.")
    tf.compat.v1.reset_default_graph()
    SCRIPT_LOGGER.info("TensorFlow default graph reset.")

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

    if enable_mixed_precision:
        try:
            policy = mixed_precision.Policy(mixed_precision_policy_name)
            mixed_precision.set_global_policy(policy)
            SCRIPT_LOGGER.info(
                f"Mixed precision policy '{mixed_precision_policy_name}' set. "
                f"Compute dtype: {policy.compute_dtype}, Variable dtype: {policy.variable_dtype}"
            )
        except Exception as e:
            SCRIPT_LOGGER.warning(
                f"Could not enable mixed precision policy '{mixed_precision_policy_name}'. Error: {e}. "
                "Training will use default precision (float32).", exc_info=True
            )
    else:
        SCRIPT_LOGGER.info("Mixed precision is DISABLED by configuration.")

# ---------------------------------------------------------------------------
# Command-Line Argument Parsing
# ---------------------------------------------------------------------------
def parse_arguments():
    """
    Parses command-line arguments for the script, allowing for flexible configuration.
    """
    parser = argparse.ArgumentParser(
        description="Run Time Series Forecasting Model (Version 3 - Standard Configuration)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Data and I/O
    parser.add_argument('--data_file', type=str, help="Path to the CSV data file.")
    parser.add_argument('--output_dir', type=str, help="Base directory for saving run outputs.")
    parser.add_argument('--time_steps', type=int, help="Number of time steps for input sequences.")

    # Training
    parser.add_argument('--epochs', type=int, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, help="Batch size for training.")

    # ModelBuilder V3 specific parameters
    # Parameters for MultiHeadAttention within residual blocks
    parser.add_argument('--num_heads_residual_block', type=int, help="Number of attention heads in residual blocks.")
    parser.add_argument('--key_dim_residual_block', type=int, help="Key dimension for attention in residual blocks.")
    # Alpha values for LeakyReLU layers
    parser.add_argument('--leaky_alpha_conv1', type=float, help="Alpha for 1st LeakyReLU in residual block's conv path.")
    parser.add_argument('--leaky_alpha_conv2', type=float, help="Alpha for 2nd LeakyReLU in residual block's conv path.")
    parser.add_argument('--leaky_alpha_after_residual_add', type=float, help="Alpha for LeakyReLU after residual sum.")
    parser.add_argument('--conv_l2_reg', type=float, help="L2 regularization for Conv1D layers in residual blocks.")
    # LSTM parameters
    parser.add_argument('--num_bilstm_layers', type=int, help="Number of BiLSTM layers.")
    parser.add_argument('--lstm_units', type=int, help="Number of units in LSTM layers.")
    # recurrent_dropout_lstm is fixed to 0.0, so no CLI argument for it.
    parser.add_argument('--lstm_l2_reg', type=float, help="L2 regularization for LSTM layers.")
    # Optimizer and output layer parameters
    parser.add_argument('--optimizer_lr', type=float, help="Learning rate for the AdamW optimizer.")
    parser.add_argument('--output_l2_reg', type=float, help="L2 regularization for the output layer.")

    # Mixed precision control
    parser.add_argument('--disable_mixed_precision', action='store_true',
                        help="Disable mixed precision training (default is enabled).")

    args = parser.parse_args()
    return args

# ---------------------------------------------------------------------------
# Main Execution Function
# ---------------------------------------------------------------------------
def main(cli_args):
    """
    Defines configurations (from defaults and CLI overrides), initializes,
    and runs the Version 3 model pipeline.
    """
    SCRIPT_LOGGER.info(f"--- Initializing Model Pipeline for Version 3 (Standard Run) ---")

    script_base_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Default Data Loading Parameters ---
    default_data_file_path = os.path.join(script_base_dir, "csv/EURUSD_Candlestick_1_Hour_BID_01.01.2010-18.02.2025.csv")
    default_time_steps = 3 # From original V3 DataLoader defaults (or adjust as needed)
    default_train_ratio = 0.96
    default_val_ratio = 0.02
    default_test_ratio = 0.02

    # --- Default Training Hyperparameters (Patterned after Version 8) ---
    default_epochs = 60
    default_batch_size = 5000 # From Version 8 for consistency

    # --- Default Model Architecture Hyperparameters for ModelBuilder Version 3 ---

    # `block_configs_v3`: Defines the structure of the convolutional residual blocks.
    # This should match the structure intended for V3. Example from original V3 Run.py:
    default_block_configs_v3 = [
        {'filters': 8, 'kernel_size': 3, 'pool_size': None},
        # Add more block configurations here if the V3 architecture uses multiple distinct residual blocks.
        # e.g., {'filters': 16, 'kernel_size': 5, 'pool_size': None}
    ]

    # `model_builder_params_v3`: Holds other architectural hyperparameters for ModelBuilder V3.
    # Keys must match arguments in ModelBuilder V3's __init__.
    # Values align with V8 where comparable, or use V3's original/sensible defaults.
    default_model_builder_params_v3 = {
        'num_heads_residual_block': 3,      # Original V3 ModelBuilder default
        'key_dim_residual_block': 4,         # Original V3 ModelBuilder default
        'leaky_relu_alpha_conv_1': 0.04,     # Original V3 ModelBuilder default
        'leaky_relu_alpha_conv_2': 0.03,     # Original V3 ModelBuilder default
        'leaky_relu_alpha_after_residual_add': 0.03, # V3 specific
        'conv_l2_reg': 0.0,

        'num_bilstm_layers': 1,              # Original V3 ModelBuilder had one BiLSTM
        'lstm_units': 200,                   # Matches V8 and original V3
        'recurrent_dropout_lstm': 0.0,       # Always off as per user request
        'lstm_l2_reg': 0.0,
        'use_batchnorm_after_lstm': False,    # Defaulting to False

        'use_batchnorm_after_final_attention': True, # As per original V3 ModelBuilder

        'use_dropout_before_output': False,  # Always off as per user request
        'dropout_rate_before_output': 0.0,
        'output_activation': 'linear',
        'output_l2_reg': 0.0,                # Matches V8 (no L2 on output)

        'optimizer_lr': 0.01,                # Matches V8
        'optimizer_weight_decay': None,
        'optimizer_clipnorm': None,
        'optimizer_clipvalue': None
    }

    # --- Override defaults with CLI arguments ---
    data_file_path = cli_args.data_file if cli_args.data_file else default_data_file_path
    time_steps = cli_args.time_steps if cli_args.time_steps is not None else default_time_steps
    epochs = cli_args.epochs if cli_args.epochs is not None else default_epochs
    batch_size = cli_args.batch_size if cli_args.batch_size is not None else default_batch_size

    # Update model_builder_params with CLI args
    model_builder_params_v3 = default_model_builder_params_v3.copy()
    # Helper to update params from CLI if provided
    def update_param(param_name, cli_value, params_dict):
        if cli_value is not None:
            params_dict[param_name] = cli_value

    update_param('optimizer_lr', cli_args.optimizer_lr, model_builder_params_v3)
    update_param('num_heads_residual_block', cli_args.num_heads_residual_block, model_builder_params_v3)
    update_param('key_dim_residual_block', cli_args.key_dim_residual_block, model_builder_params_v3)
    update_param('leaky_alpha_conv1', cli_args.leaky_alpha_conv1, model_builder_params_v3) # Corrected key
    update_param('leaky_alpha_conv2', cli_args.leaky_alpha_conv2, model_builder_params_v3) # Corrected key
    update_param('leaky_alpha_after_residual_add', getattr(cli_args, 'leaky_alpha_after_residual_add', None), model_builder_params_v3)
    update_param('conv_l2_reg', cli_args.conv_l2_reg, model_builder_params_v3)
    update_param('num_bilstm_layers', cli_args.num_bilstm_layers, model_builder_params_v3)
    update_param('lstm_units', cli_args.lstm_units, model_builder_params_v3)
    update_param('lstm_l2_reg', cli_args.lstm_l2_reg, model_builder_params_v3)
    update_param('output_l2_reg', cli_args.output_l2_reg, model_builder_params_v3)

    # block_configs_v3 is more complex for CLI, using default. Can be extended (e.g., load from JSON).
    block_configs_v3 = default_block_configs_v3

    # --- Output Directory for this Run ---
    default_output_base_dir = os.path.join(script_base_dir, "IEEE_TNNLS_Runs_V3_Standard_MP_NoDropout")
    output_base_dir = cli_args.output_dir if cli_args.output_dir else default_output_base_dir
    os.makedirs(output_base_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Log Final Configuration and Perform Sanity Checks
    # -----------------------------------------------------------------------
    SCRIPT_LOGGER.info(f"--- Final Configuration for Version 3 Run ---")
    SCRIPT_LOGGER.info(f"  Mixed Precision Enabled: {not cli_args.disable_mixed_precision}")
    SCRIPT_LOGGER.info(f"  Data file path: {data_file_path}")
    if not os.path.exists(data_file_path):
        SCRIPT_LOGGER.error(f"CRITICAL: Data file not found: {data_file_path}. Exiting.")
        sys.exit(1)

    SCRIPT_LOGGER.info(f"  Run output base directory: {output_base_dir}")
    SCRIPT_LOGGER.info(f"  Training Parameters: Epochs={epochs}, Batch Size={batch_size}")
    SCRIPT_LOGGER.info(f"  Data Processing: Time Steps={time_steps}, Train Ratio={default_train_ratio}, Val Ratio={default_val_ratio}, Test Ratio={default_test_ratio}")
    SCRIPT_LOGGER.info(f"  ModelBuilder V3 - Block Configurations: {json.dumps(block_configs_v3, indent=4)}")
    SCRIPT_LOGGER.info(f"  ModelBuilder V3 - Other Architectural Parameters:")
    for key, value in model_builder_params_v3.items():
        SCRIPT_LOGGER.info(f"    {key}: {value}")

    # -----------------------------------------------------------------------
    # Initialize and Execute the Time Series Model Pipeline
    # -----------------------------------------------------------------------
    SCRIPT_LOGGER.info("Initializing TimeSeriesModel pipeline with Version 3 configuration...")
    try:
        time_series_pipeline = TimeSeriesModel(
            file_path=data_file_path,
            time_steps=time_steps,
            train_ratio=default_train_ratio,
            val_ratio=default_val_ratio,
            test_ratio=default_test_ratio,
            base_dir=output_base_dir,
            epochs=epochs,
            batch_size=batch_size,
            block_configs=block_configs_v3, # Crucial for V3 ModelBuilder
            model_builder_params=model_builder_params_v3
        )

        SCRIPT_LOGGER.info("🚀 Starting the pipeline execution for Version 3...")
        time_series_pipeline.run()
        SCRIPT_LOGGER.info("🎉 Pipeline execution for Version 3 completed successfully.")

    except Exception as e:
        SCRIPT_LOGGER.critical(
            f"❌ A critical error occurred during the pipeline execution for Version 3: {e}",
            exc_info=True
        )
        sys.exit(1)

# ---------------------------------------------------------------------------
# Script Entry Point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_arguments()
    # Mixed precision is enabled by default unless --disable_mixed_precision is passed
    setup_environment(enable_mixed_precision=(not args.disable_mixed_precision))
    main(args)
