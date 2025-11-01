# Run.py for Version 5 (no csv/5/)
# Main script to configure, execute, and manage the time series forecasting pipeline
# for Model Version 5. This script centralizes hyperparameter settings, drawing
# structural inspiration from Version 8's Run.py for consistency and best practices.
# Key features for V5: configurable residual blocks with internal MultiHeadAttention,
# a final MultiHeadAttention layer after LSTMs, an optional intermediate Dense layer,
# and specific dtype handling for key layers.

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
# ModelBuilder for Version 5 will be instantiated by MainClass based on parameters passed.

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
        description="Run Time Series Forecasting Model (Version 5 - Standard Configuration)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Data and I/O
    parser.add_argument('--data_file', type=str, help="Path to the CSV data file.")
    parser.add_argument('--output_dir', type=str, help="Base directory for saving run outputs.")
    parser.add_argument('--time_steps', type=int, help="Number of time steps for input sequences.")

    # Training
    parser.add_argument('--epochs', type=int, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, help="Batch size for training.")

    # ModelBuilder V5 specific parameters
    # Residual Block MHA
    parser.add_argument('--num_heads_residual_block', type=int, help="Number of attention heads in residual blocks.")
    parser.add_argument('--key_dim_residual_block', type=int, help="Key dimension for attention in residual blocks.")
    # LeakyReLU Alphas
    parser.add_argument('--leaky_alpha_conv1', type=float, help="Alpha for 1st LeakyReLU in residual block's conv path.")
    parser.add_argument('--leaky_alpha_conv2', type=float, help="Alpha for 2nd LeakyReLU in residual block's conv path.")
    parser.add_argument('--leaky_alpha_after_residual_add', type=float, help="Alpha for LeakyReLU after residual sum.")
    parser.add_argument('--conv_l2_reg', type=float, help="L2 regularization for Conv1D layers in residual blocks.")
    # LSTM
    parser.add_argument('--num_bilstm_layers', type=int, help="Number of BiLSTM layers.")
    parser.add_argument('--lstm_units', type=int, help="Number of units in LSTM layers.")
    parser.add_argument('--lstm_l2_reg', type=float, help="L2 regularization for LSTM layers.")
    # Final MHA
    parser.add_argument('--num_heads_final_mha', type=int, help="Number of heads for the final MHA layer.")
    parser.add_argument('--key_dim_final_mha', type=int, help="Key dimension for the final MHA layer.")
    # Intermediate Dense (optional)
    parser.add_argument('--use_intermediate_dense', action='store_true', help="Enable the intermediate Dense layer before output.")
    parser.add_argument('--intermediate_dense_units', type=int, help="Units for the intermediate Dense layer.")
    parser.add_argument('--leaky_alpha_intermediate_dense', type=float, help="Alpha for LeakyReLU in intermediate Dense layer.")
    # Optimizer and Output
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
    and runs the Version 5 model pipeline.
    """
    SCRIPT_LOGGER.info(f"--- Initializing Model Pipeline for Version 5 (Standard Run) ---")

    script_base_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Default Data Loading Parameters ---
    default_data_file_path = os.path.join(script_base_dir, "csv/EURUSD_Candlestick_1_Hour_BID_01.01.2010-18.02.2025.csv")
    default_time_steps = 3 # From original V5 DataLoader defaults (or adjust as needed)
    default_train_ratio = 0.96
    default_val_ratio = 0.02
    default_test_ratio = 0.02

    # --- Default Training Hyperparameters (Patterned after Version 8) ---
    default_epochs = 60
    default_batch_size = 5000 # From Version 8 for consistency

    # --- Default Model Architecture Hyperparameters for ModelBuilder Version 5 ---

    # `block_configs_v5`: Defines the structure of the convolutional residual blocks.
    # Based on the original `no csv/5/Run.py`.
    default_block_configs_v5 = [
        {'filters': 8, 'kernel_size': 3, 'pool_size': None},
        # Add more block configurations here if the V5 architecture uses multiple distinct residual blocks.
    ]

    # `model_builder_params_v5`: Holds other architectural hyperparameters for ModelBuilder V5.
    # Keys must match arguments in ModelBuilder V5's __init__.
    # Values align with V8 where comparable, or use V5's original/sensible defaults.
    default_model_builder_params_v5 = {
        'num_heads_residual_block': 3,      # Original V5 ModelBuilder default
        'key_dim_residual_block': 4,         # Original V5 ModelBuilder default
        'leaky_relu_alpha_conv_1': 0.04,     # Original V5 ModelBuilder default
        'leaky_relu_alpha_conv_2': 0.03,     # Original V5 ModelBuilder default
        'leaky_relu_alpha_after_residual_add': 0.03, # Original V5 ModelBuilder default
        'conv_l2_reg': 0.0,

        'num_bilstm_layers': 1,              # Original V5 ModelBuilder had one BiLSTM
        'lstm_units': 200,                   # Matches V8 and original V5
        'recurrent_dropout_lstm': 0.0,       # Always off as per user request
        'lstm_l2_reg': 0.0,
        'use_batchnorm_after_lstm': False,    # Common practice

        'num_heads_final_mha': 6,           # Original V5 ModelBuilder default for final MHA
        'key_dim_final_mha': 10,              # Original V5 ModelBuilder default for final MHA
        'use_batchnorm_after_final_mha': True, # As per original V5 ModelBuilder

        'use_intermediate_dense': False,     # Original V5 had this commented out
        'intermediate_dense_units': 256,     # Default if enabled (from original V5 commented code)
        'leaky_relu_alpha_intermediate_dense': 0.00, # Original V5 commented code
        'intermediate_dense_l2_reg': 0.0,
        'use_batchnorm_intermediate_dense': True, # Original V5 commented code had BN

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
    model_builder_params_v5 = default_model_builder_params_v5.copy()
    def update_param(param_name, cli_value, params_dict): # Helper
        if cli_value is not None:
            params_dict[param_name] = cli_value

    update_param('optimizer_lr', cli_args.optimizer_lr, model_builder_params_v5)
    update_param('num_heads_residual_block', cli_args.num_heads_residual_block, model_builder_params_v5)
    update_param('key_dim_residual_block', cli_args.key_dim_residual_block, model_builder_params_v5)
    update_param('leaky_relu_alpha_conv_1', cli_args.leaky_alpha_conv1, model_builder_params_v5)
    update_param('leaky_relu_alpha_conv_2', cli_args.leaky_alpha_conv2, model_builder_params_v5)
    update_param('leaky_relu_alpha_after_residual_add', getattr(cli_args, 'leaky_alpha_after_residual_add', None), model_builder_params_v5)
    update_param('conv_l2_reg', cli_args.conv_l2_reg, model_builder_params_v5)
    update_param('num_bilstm_layers', cli_args.num_bilstm_layers, model_builder_params_v5)
    update_param('lstm_units', cli_args.lstm_units, model_builder_params_v5)
    update_param('lstm_l2_reg', cli_args.lstm_l2_reg, model_builder_params_v5)
    update_param('num_heads_final_mha', getattr(cli_args, 'num_heads_final_mha', None), model_builder_params_v5)
    update_param('key_dim_final_mha', getattr(cli_args, 'key_dim_final_mha', None), model_builder_params_v5)
    if cli_args.use_intermediate_dense: # This is a boolean flag
        model_builder_params_v5['use_intermediate_dense'] = True
    update_param('intermediate_dense_units', getattr(cli_args, 'intermediate_dense_units', None), model_builder_params_v5)
    update_param('leaky_alpha_intermediate_dense', getattr(cli_args, 'leaky_alpha_intermediate_dense', None), model_builder_params_v5)
    update_param('output_l2_reg', cli_args.output_l2_reg, model_builder_params_v5)

    block_configs_v5 = default_block_configs_v5

    default_output_base_dir = os.path.join(script_base_dir, "IEEE_TNNLS_Runs_V5_Standard_MP_NoDropout")
    output_base_dir = cli_args.output_dir if cli_args.output_dir else default_output_base_dir
    os.makedirs(output_base_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Log Final Configuration and Perform Sanity Checks
    # -----------------------------------------------------------------------
    SCRIPT_LOGGER.info(f"--- Final Configuration for Version 5 Run ---")
    SCRIPT_LOGGER.info(f"  Mixed Precision Enabled: {not cli_args.disable_mixed_precision}")
    SCRIPT_LOGGER.info(f"  Data file path: {data_file_path}")
    if not os.path.exists(data_file_path):
        SCRIPT_LOGGER.error(f"CRITICAL: Data file not found: {data_file_path}. Exiting.")
        sys.exit(1)

    SCRIPT_LOGGER.info(f"  Run output base directory: {output_base_dir}")
    SCRIPT_LOGGER.info(f"  Training Parameters: Epochs={epochs}, Batch Size={batch_size}")
    SCRIPT_LOGGER.info(f"  Data Processing: Time Steps={time_steps}, Train Ratio={default_train_ratio}, Val Ratio={default_val_ratio}, Test Ratio={default_test_ratio}")
    SCRIPT_LOGGER.info(f"  ModelBuilder V5 - Block Configurations: {json.dumps(block_configs_v5, indent=4)}")
    SCRIPT_LOGGER.info(f"  ModelBuilder V5 - Other Architectural Parameters:")
    for key, value in model_builder_params_v5.items():
        SCRIPT_LOGGER.info(f"    {key}: {value}")

    # -----------------------------------------------------------------------
    # Initialize and Execute the Time Series Model Pipeline
    # -----------------------------------------------------------------------
    SCRIPT_LOGGER.info("Initializing TimeSeriesModel pipeline with Version 5 configuration...")
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
            block_configs=block_configs_v5, # Crucial for V5 ModelBuilder
            model_builder_params=model_builder_params_v5
        )

        SCRIPT_LOGGER.info("🚀 Starting the pipeline execution for Version 5...")
        time_series_pipeline.run()
        SCRIPT_LOGGER.info("🎉 Pipeline execution for Version 5 completed successfully.")

    except Exception as e:
        SCRIPT_LOGGER.critical(
            f"❌ A critical error occurred during the pipeline execution for Version 5: {e}",
            exc_info=True
        )
        sys.exit(1)

# ---------------------------------------------------------------------------
# Script Entry Point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_arguments()
    setup_environment(enable_mixed_precision=(not args.disable_mixed_precision))
    main(args)
