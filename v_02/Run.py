# /no_csv_2/2/Run.py

import os
import sys
import logging
import argparse
import json
import tensorflow as tf
from tensorflow.keras import mixed_precision, backend as K
from MainClass import TimeSeriesModel

# --- Constants and Initial Setup ---
SEED = 42
tf.random.set_seed(SEED)
SCRIPT_LOGGER = logging.getLogger(__name__)

def setup_environment(enable_mixed_precision: bool = True):
    """Configures the global execution environment for the pipeline."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s (%(filename)s:%(lineno)d)', datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
    SCRIPT_LOGGER.info(f"Global TensorFlow seed set to {SEED} for reproducibility.")
    K.clear_session()
    SCRIPT_LOGGER.info("Keras session cleared.")
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            SCRIPT_LOGGER.info("GPU memory growth enabled.")
            if enable_mixed_precision:
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                SCRIPT_LOGGER.info("Mixed precision policy 'mixed_float16' enabled.")
        except RuntimeError as e: SCRIPT_LOGGER.error(f"GPU setup failed: {e}", exc_info=True)
    else: SCRIPT_LOGGER.info("No GPU detected.")

def parse_arguments():
    """Parses command-line arguments to configure the pipeline run."""
    parser = argparse.ArgumentParser(description="Run Time Series Forecasting Model (Version 2 - Standardized)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', type=str, help="Path to the input CSV data file.")
    parser.add_argument('--output_dir', type=str, help="Base directory for saving run outputs.")
    parser.add_argument('--time_steps', type=int, help="Look-back window size.")
    parser.add_argument('--epochs', type=int, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, help="Batch size for training.")
    parser.add_argument('--optimizer_lr', type=float, help="Learning rate for the AdamW optimizer.")
    parser.add_argument('--disable_mixed_precision', action='store_true', help="Disable mixed precision training.")
    return parser.parse_args()

def main(cli_args):
    """Main execution function for the Version 2 pipeline."""
    SCRIPT_LOGGER.info("--- Initializing Pipeline for Model Version 2 ---")
    
    script_base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- Default Configurations for Version 2 ---
    default_data_file_path = os.path.join(script_base_dir, "csv/EURUSD_Candlestick_1_Hour_BID_01.01.2010-18.02.2025.csv")
    default_time_steps = 3
    default_train_ratio = 0.96
    default_val_ratio = 0.02
    default_test_ratio = 0.02
    default_epochs = 60
    default_batch_size = 5000

    # Block configurations define the convolutional layers.
    default_block_configs = [{'filters': 8, 'kernel_size': 3, 'pool_size': None}]
    
    # These parameters are specific to ModelBuilder version 2
    default_model_params = {
        'num_heads_conv_block': 3,
        'key_dim_conv_block': 4,
        'leaky_relu_alpha_conv_1': 0.04,
        'leaky_relu_alpha_conv_2': 0.03,
        'conv_l2_reg': 0.0,
        'num_bilstm_layers': 1,
        'lstm_units': 200,
        'recurrent_dropout_lstm': 0.0,
        'lstm_l2_reg': 0.0,
        'use_batchnorm_after_final_attention': True,
        'use_dropout_before_output': False,
        'dropout_rate_before_output': 0.0,
        'output_activation': 'linear',
        'output_l2_reg': 0.0,
        'optimizer_lr': 0.01,
        'optimizer_weight_decay': None,
        'optimizer_clipnorm': None,
        'optimizer_clipvalue': None
    }

    # --- Override with CLI arguments ---
    data_file_path = cli_args.data_file or default_data_file_path
    time_steps = cli_args.time_steps or default_time_steps
    epochs = cli_args.epochs or default_epochs
    batch_size = cli_args.batch_size or default_batch_size
    
    model_builder_params = default_model_params.copy()
    if cli_args.optimizer_lr:
        model_builder_params['optimizer_lr'] = cli_args.optimizer_lr
    
    output_dir = cli_args.output_dir or os.path.join(script_base_dir, "IEEE_TNNLS_Runs_V2")
    os.makedirs(output_dir, exist_ok=True)

    # --- Comprehensive Final Configuration Logging ---
    SCRIPT_LOGGER.info("--- Final Configuration for Version 2 Run ---")
    SCRIPT_LOGGER.info(f"  Mixed Precision Enabled: {not cli_args.disable_mixed_precision}")
    SCRIPT_LOGGER.info(f"  Data file path: {data_file_path}")
    if not os.path.exists(data_file_path):
        SCRIPT_LOGGER.error(f"CRITICAL: Data file not found: {data_file_path}. Exiting.")
        sys.exit(1)
    SCRIPT_LOGGER.info(f"  Run output base directory: {output_dir}")
    SCRIPT_LOGGER.info(f"  Training: Epochs={epochs}, Batch Size={batch_size}")
    SCRIPT_LOGGER.info(f"  Data Processing: Time Steps={time_steps}, Train Ratio={default_train_ratio}, Val Ratio={default_val_ratio}, Test Ratio={default_test_ratio}")
    SCRIPT_LOGGER.info(f"  ModelBuilder V2 - Block Configurations: {json.dumps(default_block_configs, indent=4)}")
    SCRIPT_LOGGER.info(f"  ModelBuilder V2 - Other Architectural Hyperparameters:")
    for key, value in model_builder_params.items():
        SCRIPT_LOGGER.info(f"    {key}: {value}")

    # --- Execute Pipeline ---
    try:
        pipeline = TimeSeriesModel(
            file_path=data_file_path, time_steps=time_steps,
            train_ratio=default_train_ratio, val_ratio=default_val_ratio, test_ratio=default_test_ratio,
            base_dir=output_dir, epochs=epochs, batch_size=batch_size,
            block_configs=default_block_configs, model_builder_params=model_builder_params
        )
        pipeline.run()
        SCRIPT_LOGGER.info("🎉 Version 2 pipeline completed successfully.")
    except Exception as e:
        SCRIPT_LOGGER.critical(f"❌ Version 2 pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    args = parse_arguments()
    setup_environment(enable_mixed_precision=(not args.disable_mixed_precision))
    main(args)