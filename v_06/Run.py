# Run.py for Version 6 (no csv/6/)
# Main script to configure, execute, and manage the time series forecasting pipeline
# for Model Version 6. This script centralizes hyperparameter settings, drawing
# structural inspiration from Version 8's Run.py for consistency and best practices.
# V6 Architecture: Residual Blocks (Conv1D+MHA) -> BiLSTM -> MHA (V6-specific config) -> Dense -> Output

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
# ModelBuilder for Version 6 (refactored) will be instantiated by MainClass.

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
def setup_environment(enable_mixed_precision: bool = True, # Defaulting to True
                      mixed_precision_policy_name: str = 'mixed_float16'):
    """
    Configures the global TensorFlow environment, GPU settings, and basic application logging.
    This function should be consistent across all Run.py scripts.
    """
    # Configure base logging for the application (console output).
    logging.basicConfig(
        level=logging.INFO, # Set to logging.DEBUG for more verbose output.
        format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s (%(filename)s:%(lineno)d)',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout) # Directs logs to the console.
        ]
    )
    SCRIPT_LOGGER.info(f"Global TensorFlow random seed set to {SEED} for potential reproducibility.")

    # Clear Keras session to ensure a clean state.
    K.clear_session()
    SCRIPT_LOGGER.info("Keras session cleared.")

    # Reset TensorFlow's default graph (more relevant for TF1 or complex TF2 scenarios).
    tf.compat.v1.reset_default_graph()
    SCRIPT_LOGGER.info("TensorFlow default graph reset.")

    # GPU memory management: Enable memory growth for each detected GPU.
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
    else:
        SCRIPT_LOGGER.info("Mixed precision is DISABLED by configuration.")

# ---------------------------------------------------------------------------
# Command-Line Argument Parsing
# ---------------------------------------------------------------------------
def parse_arguments():
    """
    Parses command-line arguments for the script.
    Allows for overriding default configurations for experiments without code changes.
    """
    parser = argparse.ArgumentParser(
        description="Run Time Series Forecasting Model (Version 6 - Standardized Hyperparameters)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    # Data and I/O arguments
    parser.add_argument('--data_file', type=str, help="Path to the CSV data file.")
    parser.add_argument('--output_dir', type=str, help="Base directory for saving run outputs.")
    parser.add_argument('--time_steps', type=int, help="Number of time steps (look-back window) for input sequences.")

    # Training arguments
    parser.add_argument('--epochs', type=int, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, help="Batch size for training.")

    # Model architecture arguments (specific to ModelBuilder V6, standardized from V8)
    # Residual Block MHA
    parser.add_argument('--num_heads_res_block', type=int, help="Heads for MHA in residual blocks.")
    parser.add_argument('--key_dim_res_block', type=int, help="Key dim for MHA in residual blocks.")
    # LeakyReLU Alphas
    parser.add_argument('--leaky_alpha_conv1_res_block', type=float, help="Alpha for 1st LeakyReLU in res block conv path.")
    parser.add_argument('--leaky_alpha_conv2_res_block', type=float, help="Alpha for 2nd LeakyReLU in res block conv path.")
    parser.add_argument('--leaky_alpha_after_add_res_block', type=float, help="Alpha for LeakyReLU after res sum.")
    parser.add_argument('--conv_l2_reg_res_block', type=float, help="L2 reg for Conv1D in res blocks.")
    # LSTM
    parser.add_argument('--num_bilstm_layers', type=int, help="Number of BiLSTM layers.")
    parser.add_argument('--lstm_units', type=int, help="Units in LSTM layers.")
    parser.add_argument('--lstm_l2_reg', type=float, help="L2 reg for LSTM layers.")
    # Dense layer after flatten
    parser.add_argument('--dense_units_after_flatten', type=int, help="Units for Dense layer after flatten.")
    parser.add_argument('--leaky_alpha_dense_after_flatten', type=float, help="Alpha for LeakyReLU after flatten-dense.")
    parser.add_argument('--dense_l2_reg_after_flatten', type=float, help="L2 reg for Dense layer after flatten.")
    # Optimizer and Output
    parser.add_argument('--optimizer_lr', type=float, help="Learning rate for AdamW.")
    parser.add_argument('--output_l2_reg', type=float, help="L2 reg for the output layer.")

    # Boolean flags
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
    and runs the Version 6 model pipeline.
    """
    SCRIPT_LOGGER.info(f"--- Initializing Model Pipeline for Version 6 (Standardized Run) ---")

    script_base_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Default Data Loading Parameters (Standardized with V8) ---
    default_data_file_path = os.path.join(script_base_dir, "csv/EURUSD_Candlestick_1_Hour_BID_01.01.2010-18.02.2025.csv")
    default_time_steps = 3       # Standardized with V8
    default_train_ratio = 0.96
    default_val_ratio = 0.02
    default_test_ratio = 0.02

    # --- Default Training Hyperparameters (Standardized with V8) ---
    default_epochs = 60
    default_batch_size = 5000

    # --- Default Model Architecture: Block Configurations for V6 (Standardized) ---
    # V6's architecture uses residual blocks. The number of blocks is defined by this list.
    # The internal parameters (filters, kernel_size, pool_size) are standardized from V8's block.
    default_block_configs_v6 = [
        {'filters': 8, 'kernel_size': 3, 'pool_size': None}, # Standardized from V8's single block
        # If V6's unique architecture requires more residual blocks, add their configs here,
        # but use the standardized filter/kernel/pool values.
    ]

    # --- Default Model Architecture: Other Hyperparameters for ModelBuilder V6 (Standardized from V8) ---
    default_model_builder_params_v6 = {
        'num_heads_res_block': 3,
        'key_dim_res_block': 4,
        'leaky_relu_alpha_conv1_res_block': 0.04, # V8's leaky_relu_alpha_res_block
        'leaky_relu_alpha_conv2_res_block': 0.03, # V8's leaky_relu_alpha_res_block2 (for 2nd conv)
        'leaky_relu_alpha_after_add_res_block': 0.03, # V8's leaky_relu_alpha_res_block2 (for after Add)
        'conv_l2_reg_res_block': 0.0, # Mapped from V8's general l2_reg
        
        'num_bilstm_layers': 1, 
        'lstm_units': 200,
        'recurrent_dropout_lstm': 0.0, # Standardized: OFF
        'lstm_l2_reg': 0.0, # Mapped from V8's general l2_reg
        'use_batchnorm_after_lstm': False, # Good practice

        'use_batchnorm_after_post_lstm_mha': True, # V6 specific architectural choice

        'dense_units_after_flatten': 256, # V8's dense_units_before_output
        'leaky_relu_alpha_dense_after_flatten': 0.00, # V8's leaky_relu_alpha_dense
        'dense_l2_reg_after_flatten': 0.0, # Mapped from V8's general l2_reg
        
        'output_activation': 'linear',
        'output_l2_reg': 0.0, # V8's general l2_reg
        
        'optimizer_lr': 0.01,
        'optimizer_weight_decay': None, # From V8
        'optimizer_clipnorm': None,     # From V8
        'optimizer_clipvalue': None,    # From V8
    }

    # --- Override defaults with CLI arguments if provided ---
    data_file_path = cli_args.data_file if cli_args.data_file else default_data_file_path
    time_steps = cli_args.time_steps if cli_args.time_steps is not None else default_time_steps
    epochs = cli_args.epochs if cli_args.epochs is not None else default_epochs
    batch_size = cli_args.batch_size if cli_args.batch_size is not None else default_batch_size

    block_configs_v6 = default_block_configs_v6 # Using default, can be made CLI configurable if needed (e.g., via JSON string)
    
    model_builder_params_v6 = default_model_builder_params_v6.copy()
    # Update model_builder_params with CLI args for specific model hyperparameters
    # This allows fine-tuning specific parameters if needed for a particular run, overriding the defaults.
    if cli_args.optimizer_lr is not None: model_builder_params_v6['optimizer_lr'] = cli_args.optimizer_lr
    if cli_args.num_heads_res_block is not None: model_builder_params_v6['num_heads_res_block'] = cli_args.num_heads_res_block
    if cli_args.key_dim_res_block is not None: model_builder_params_v6['key_dim_res_block'] = cli_args.key_dim_res_block
    if cli_args.leaky_alpha_conv1_res_block is not None: model_builder_params_v6['leaky_relu_alpha_conv1_res_block'] = cli_args.leaky_alpha_conv1_res_block
    if cli_args.leaky_alpha_conv2_res_block is not None: model_builder_params_v6['leaky_relu_alpha_conv2_res_block'] = cli_args.leaky_alpha_conv2_res_block
    if cli_args.leaky_alpha_after_add_res_block is not None: model_builder_params_v6['leaky_relu_alpha_after_add_res_block'] = cli_args.leaky_alpha_after_add_res_block
    if cli_args.conv_l2_reg_res_block is not None: model_builder_params_v6['conv_l2_reg_res_block'] = cli_args.conv_l2_reg_res_block
    if cli_args.num_bilstm_layers is not None: model_builder_params_v6['num_bilstm_layers'] = cli_args.num_bilstm_layers
    if cli_args.lstm_units is not None: model_builder_params_v6['lstm_units'] = cli_args.lstm_units
    if cli_args.lstm_l2_reg is not None: model_builder_params_v6['lstm_l2_reg'] = cli_args.lstm_l2_reg
    if cli_args.dense_units_after_flatten is not None: model_builder_params_v6['dense_units_after_flatten'] = cli_args.dense_units_after_flatten
    if cli_args.leaky_alpha_dense_after_flatten is not None: model_builder_params_v6['leaky_relu_alpha_dense_after_flatten'] = cli_args.leaky_alpha_dense_after_flatten
    if cli_args.dense_l2_reg_after_flatten is not None: model_builder_params_v6['dense_l2_reg_after_flatten'] = cli_args.dense_l2_reg_after_flatten
    if cli_args.output_l2_reg is not None: model_builder_params_v6['output_l2_reg'] = cli_args.output_l2_reg
    
    # --- Output Directory for this Run ---
    default_output_base_dir = os.path.join(script_base_dir, "IEEE_TNNLS_Runs_V6_Standardized")
    output_base_dir = cli_args.output_dir if cli_args.output_dir else default_output_base_dir
    os.makedirs(output_base_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Log Final Configuration and Perform Sanity Checks
    # -----------------------------------------------------------------------
    SCRIPT_LOGGER.info(f"--- Final Configuration for Version 6 Run ---")
    SCRIPT_LOGGER.info(f"  Mixed Precision Enabled: {not cli_args.disable_mixed_precision}")
    SCRIPT_LOGGER.info(f"  Data file path: {data_file_path}")
    if not os.path.exists(data_file_path):
        SCRIPT_LOGGER.error(f"CRITICAL: Data file not found: {data_file_path}. Please verify the path. Exiting.")
        sys.exit(1)

    SCRIPT_LOGGER.info(f"  Run output base directory: {output_base_dir}")
    SCRIPT_LOGGER.info(f"  Training Parameters: Epochs={epochs}, Batch Size={batch_size}")
    SCRIPT_LOGGER.info(f"  Data Processing Parameters: Time Steps={time_steps}, Train Ratio={default_train_ratio}, Val Ratio={default_val_ratio}, Test Ratio={default_test_ratio}")
    SCRIPT_LOGGER.info(f"  ModelBuilder V6 - Architectural Block Configurations: {json.dumps(block_configs_v6, indent=4)}")
    SCRIPT_LOGGER.info(f"  ModelBuilder V6 - Other Architectural Hyperparameters:")
    for key, value in model_builder_params_v6.items():
        SCRIPT_LOGGER.info(f"    {key}: {value}")

    # -----------------------------------------------------------------------
    # Initialize and Execute the Time Series Model Pipeline
    # -----------------------------------------------------------------------
    SCRIPT_LOGGER.info("Initializing TimeSeriesModel pipeline with Version 6 configuration...")
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
            block_configs=block_configs_v6, # Pass V6's architectural block definition
            model_builder_params=model_builder_params_v6 # Pass standardized HPs for V6 ModelBuilder
        )

        SCRIPT_LOGGER.info("🚀 Starting the pipeline execution for Version 6...")
        time_series_pipeline.run()
        SCRIPT_LOGGER.info("🎉 Pipeline execution for Version 6 completed successfully.")

    except Exception as e:
        SCRIPT_LOGGER.critical(
            f"❌ A critical error occurred during the pipeline execution for Version 6: {e}",
            exc_info=True # This will include the full stack trace in the log.
        )
        sys.exit(1) # Indicate failure to the calling environment.

# ---------------------------------------------------------------------------
# Script Entry Point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    cli_args = parse_arguments()
    # Mixed precision is enabled by default unless --disable_mixed_precision is passed
    setup_environment(enable_mixed_precision=(not cli_args.disable_mixed_precision))
    main(cli_args)
