# Run.py for Version 7 (no csv/7/)
# Main script to configure, execute, and manage the time series forecasting pipeline
# for Model Version 7. This script centralizes hyperparameter settings, drawing
# structural inspiration from Version 8's Run.py for consistency and best practices.
# V7 Architecture: Residual Blocks (Conv1D+MHA, V7 dual pooling) -> BiLSTM -> 
#                  MHA (V7-specific config) -> MixOfExperts -> Flatten -> Output Dense.

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

# ---------------------------------------------------------------------------
# Script Constants and Configuration
# ---------------------------------------------------------------------------
SEED = 42 # Seed for reproducibility
tf.random.set_seed(SEED)
# import numpy as np; np.random.seed(SEED) # Uncomment if NumPy's random functions are used directly
# import random; random.seed(SEED) # Uncomment if Python's random module is used directly

SCRIPT_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment Setup Function
# ---------------------------------------------------------------------------
def setup_environment(enable_mixed_precision: bool = True, 
                      mixed_precision_policy_name: str = 'mixed_float16'):
    """
    Configures the global TensorFlow environment, GPU settings, and basic application logging.
    This function should be consistent across all Run.py scripts.
    """
    # Configure base logging for the application (console output).
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
        description="Run Time Series Forecasting Model (Version 7 - Standardized Hyperparameters)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    # Data and I/O arguments
    parser.add_argument('--data_file', type=str, help="Path to the CSV data file.")
    parser.add_argument('--output_dir', type=str, help="Base directory for saving run outputs.")
    parser.add_argument('--time_steps', type=int, help="Number of time steps (look-back window) for input sequences.")

    # Training arguments
    parser.add_argument('--epochs', type=int, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, help="Batch size for training.")

    # ModelBuilder V7 specific parameters (standardized from V8 where applicable)
    # Residual Block parameters
    parser.add_argument('--num_heads_res_block', type=int, help="Number of attention heads in residual blocks.")
    parser.add_argument('--key_dim_res_block', type=int, help="Key dimension for attention in residual blocks.")
    parser.add_argument('--leaky_alpha_conv1_res_block', type=float, help="Alpha for 1st LeakyReLU in res block conv path.")
    parser.add_argument('--leaky_alpha_conv2_res_block', type=float, help="Alpha for 2nd LeakyReLU in res block conv path.")
    parser.add_argument('--leaky_alpha_after_add_res_block', type=float, help="Alpha for LeakyReLU after residual sum.")
    parser.add_argument('--conv_l2_reg_res_block', type=float, help="L2 regularization for Conv1D layers in residual blocks.")
    # LSTM parameters
    parser.add_argument('--num_bilstm_layers', type=int, help="Number of BiLSTM layers.")
    parser.add_argument('--lstm_units', type=int, help="Number of units in LSTM layers.")
    parser.add_argument('--lstm_l2_reg', type=float, help="L2 regularization for LSTM layers.")
    # MixOfExperts parameters
    parser.add_argument('--moe_num_experts', type=int, help="Number of experts in MoE layer.")
    parser.add_argument('--moe_units', type=int, help="Units for each expert in MoE layer.")
    parser.add_argument('--moe_leaky_relu_alpha', type=float, help="Alpha for LeakyReLU in MoE experts.")
    # Optimizer and Output Layer parameters
    parser.add_argument('--optimizer_lr', type=float, help="Learning rate for the AdamW optimizer.")
    parser.add_argument('--output_l2_reg', type=float, help="L2 regularization for the output layer.")

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
    and runs the Version 7 model pipeline.
    """
    SCRIPT_LOGGER.info(f"--- Initializing Model Pipeline for Version 7 (Standardized Run) ---")

    script_base_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Default Data Loading Parameters (Standardized with V8) ---
    default_data_file_path = os.path.join(script_base_dir, "csv/EURUSD_Candlestick_1_Hour_BID_01.01.2010-18.02.2025.csv")
    default_time_steps = 3       # Standardized with V8 (original V7 also used 3)
    default_train_ratio = 0.96
    default_val_ratio = 0.02
    default_test_ratio = 0.02

    # --- Default Training Hyperparameters (Standardized with V8) ---
    default_epochs = 60
    default_batch_size = 5000

    # --- Default Model Architecture: Block Configurations for V7 ---
    # V7's architecture uses residual blocks. The number of blocks is defined by this list.
    # Filters and kernel_size are standardized from V8.
    # pool_size is an ARCHITECTURAL CHOICE for V7's dual pooling within _residual_block.
    # Original V7 Run.py had pool_size: 2. This is preserved here.
    default_block_configs_v7 = [
        {'filters': 8, 'kernel_size': 3, 'pool_size': None}, 
        # If V7's unique architecture requires more residual blocks, define them here,
        # using the standardized filter/kernel values and V7's specific pool_size.
    ]

    # --- Default Model Architecture: Other Hyperparameters for ModelBuilder V7 (Standardized from V8) ---
    # These parameters are passed to the refactored ModelBuilder V7.
    # Parameters for the removed intermediate Dense layer are NOT included.
    default_model_builder_params_v7 = {
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

        'use_batchnorm_after_post_lstm_mha': True, # V7 specific architectural choice

        'moe_num_experts': 12, # From V8 template (if V8 defines it, else V7 original)
        'moe_units': 64,       # From V8 template (if V8 defines it, else V7 original)
        'moe_leaky_relu_alpha': 0.01, # V7 original default for MoE experts' LeakyReLU
        'use_batchnorm_after_moe': True, # V7 specific architectural choice
        
        # V7 has no intermediate Dense layer after Flatten/MoE before the final output.
        # The 'dense_units_after_flatten', 'leaky_relu_alpha_dense_after_flatten', etc. 
        # from the previous incorrect V7 ModelBuilder are removed here.

        'output_activation': 'linear',
        'output_l2_reg': 0.0, # V8's general l2_reg
        
        'optimizer_lr': 0.01,
        'optimizer_weight_decay': None, 
        'optimizer_clipnorm': None,     
        'optimizer_clipvalue': None,    
    }

    # --- Override defaults with CLI arguments if provided ---
    data_file_path = cli_args.data_file if cli_args.data_file else default_data_file_path
    time_steps = cli_args.time_steps if cli_args.time_steps is not None else default_time_steps
    epochs = cli_args.epochs if cli_args.epochs is not None else default_epochs
    batch_size = cli_args.batch_size if cli_args.batch_size is not None else default_batch_size

    block_configs_v7 = default_block_configs_v7 
    
    model_builder_params_v7 = default_model_builder_params_v7.copy()
    # Update model_builder_params with CLI args for specific model hyperparameters
    if cli_args.optimizer_lr is not None: model_builder_params_v7['optimizer_lr'] = cli_args.optimizer_lr
    if cli_args.num_heads_res_block is not None: model_builder_params_v7['num_heads_res_block'] = cli_args.num_heads_res_block
    if cli_args.key_dim_res_block is not None: model_builder_params_v7['key_dim_res_block'] = cli_args.key_dim_res_block
    if cli_args.leaky_alpha_conv1_res_block is not None: model_builder_params_v7['leaky_relu_alpha_conv1_res_block'] = cli_args.leaky_alpha_conv1_res_block
    if cli_args.leaky_alpha_conv2_res_block is not None: model_builder_params_v7['leaky_relu_alpha_conv2_res_block'] = cli_args.leaky_alpha_conv2_res_block
    if cli_args.leaky_alpha_after_add_res_block is not None: model_builder_params_v7['leaky_relu_alpha_after_add_res_block'] = cli_args.leaky_alpha_after_add_res_block
    if cli_args.conv_l2_reg_res_block is not None: model_builder_params_v7['conv_l2_reg_res_block'] = cli_args.conv_l2_reg_res_block
    if cli_args.num_bilstm_layers is not None: model_builder_params_v7['num_bilstm_layers'] = cli_args.num_bilstm_layers
    if cli_args.lstm_units is not None: model_builder_params_v7['lstm_units'] = cli_args.lstm_units
    if cli_args.lstm_l2_reg is not None: model_builder_params_v7['lstm_l2_reg'] = cli_args.lstm_l2_reg
    if cli_args.moe_num_experts is not None: model_builder_params_v7['moe_num_experts'] = cli_args.moe_num_experts
    if cli_args.moe_units is not None: model_builder_params_v7['moe_units'] = cli_args.moe_units
    if cli_args.moe_leaky_relu_alpha is not None: model_builder_params_v7['moe_leaky_relu_alpha'] = cli_args.moe_leaky_relu_alpha
    if cli_args.output_l2_reg is not None: model_builder_params_v7['output_l2_reg'] = cli_args.output_l2_reg
        
    default_output_base_dir = os.path.join(script_base_dir, "IEEE_TNNLS_Runs_V7_Standardized")
    output_base_dir = cli_args.output_dir if cli_args.output_dir else default_output_base_dir
    os.makedirs(output_base_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Log Final Configuration
    # -----------------------------------------------------------------------
    SCRIPT_LOGGER.info(f"--- Final Configuration for Version 7 Run ---")
    SCRIPT_LOGGER.info(f"  Mixed Precision Enabled: {not cli_args.disable_mixed_precision}")
    SCRIPT_LOGGER.info(f"  Data file path: {data_file_path}")
    if not os.path.exists(data_file_path):
        SCRIPT_LOGGER.error(f"CRITICAL: Data file not found: {data_file_path}. Exiting.")
        sys.exit(1)
    SCRIPT_LOGGER.info(f"  Run output base directory: {output_base_dir}")
    SCRIPT_LOGGER.info(f"  Training: Epochs={epochs}, Batch Size={batch_size}")
    SCRIPT_LOGGER.info(f"  Data Processing: Time Steps={time_steps}, Train Ratio={default_train_ratio}, Val Ratio={default_val_ratio}, Test Ratio={default_test_ratio}")
    SCRIPT_LOGGER.info(f"  ModelBuilder V7 - Block Configurations: {json.dumps(block_configs_v7, indent=4)}")
    SCRIPT_LOGGER.info(f"  ModelBuilder V7 - Other Architectural Hyperparameters:")
    for key, value in model_builder_params_v7.items():
        SCRIPT_LOGGER.info(f"    {key}: {value}")

    # -----------------------------------------------------------------------
    # Initialize and Execute Pipeline
    # -----------------------------------------------------------------------
    SCRIPT_LOGGER.info("Initializing TimeSeriesModel pipeline with Version 7 configuration...")
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
            block_configs=block_configs_v7, # Pass V7's architectural block definition
            model_builder_params=model_builder_params_v7 # Pass standardized HPs for V7 ModelBuilder
        )
        SCRIPT_LOGGER.info("🚀 Starting the pipeline execution for Version 7...")
        time_series_pipeline.run()
        SCRIPT_LOGGER.info("🎉 Pipeline execution for Version 7 completed successfully.")
    except Exception as e:
        SCRIPT_LOGGER.critical(f"❌ Critical error during Version 7 pipeline execution: {e}", exc_info=True)
        sys.exit(1) # Indicate failure to the calling environment.

# ---------------------------------------------------------------------------
# Script Entry Point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    cli_args_parsed = parse_arguments()
    # Mixed precision is enabled by default unless --disable_mixed_precision is passed
    setup_environment(enable_mixed_precision=(not cli_args_parsed.disable_mixed_precision))
    main(cli_args_parsed)
