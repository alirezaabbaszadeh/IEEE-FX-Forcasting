# /no_csv_1/10/Run.py

import os
import sys
import logging
import argparse
import json
from dataclasses import asdict, replace
import tensorflow as tf
from tensorflow.keras import mixed_precision, backend as K
from MainClass import TimeSeriesModel
from config import (
    DataParameters,
    TrainingParameters,
    ModelBuilderConfig,
    PipelineConfig,
)

# --- Constants and Initial Setup ---
SEED = 42
tf.random.set_seed(SEED)
SCRIPT_LOGGER = logging.getLogger(__name__)

def setup_environment(enable_mixed_precision: bool = True):
    """Configures the global execution environment for the pipeline."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s (%(filename)s:%(lineno)d)', datefmt='%Y-%m-%d %H:%M:%S')
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
    parser = argparse.ArgumentParser(description="Run Time Series Forecasting Model (Version 10 - Standardized)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', type=str, help="Path to the input CSV data file.")
    parser.add_argument('--output_dir', type=str, help="Base directory for saving run outputs.")
    parser.add_argument('--time_steps', type=int, help="Look-back window size.")
    parser.add_argument('--epochs', type=int, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, help="Batch size for training.")
    parser.add_argument('--optimizer_lr', type=float, help="Learning rate for the optimizer.")
    parser.add_argument('--disable_mixed_precision', action='store_true', help="Disable mixed precision training.")
    return parser.parse_args()

def main(cli_args):
    """Main execution function for the Version 10 pipeline."""
    SCRIPT_LOGGER.info("--- Initializing Pipeline for Model Version 10 ---")
    
    script_base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- Default Configurations for Version 10 (Corrected and Standardized) ---
    default_data_config = DataParameters(
        file_path=os.path.join(
            script_base_dir, "csv/EURUSD_Candlestick_1_Hour_BID_01.01.2010-18.02.2025.csv"
        ),
        time_steps=3,
        train_ratio=0.96,
        val_ratio=0.02,
        test_ratio=0.02,
    )
    default_training_config = TrainingParameters(epochs=60, batch_size=5000)
    default_model_builder_config = ModelBuilderConfig()

    # --- Override with CLI arguments ---
    data_config = replace(
        default_data_config,
        file_path=cli_args.data_file or default_data_config.file_path,
        time_steps=cli_args.time_steps or default_data_config.time_steps,
    )

    training_config = replace(
        default_training_config,
        epochs=cli_args.epochs or default_training_config.epochs,
        batch_size=cli_args.batch_size or default_training_config.batch_size,
    )

    model_builder_config = ModelBuilderConfig(**asdict(default_model_builder_config))
    if cli_args.optimizer_lr:
        model_builder_config = replace(
            model_builder_config, optimizer_lr=cli_args.optimizer_lr
        )

    output_dir = cli_args.output_dir or os.path.join(script_base_dir, "IEEE_TNNLS_Runs_V10")
    os.makedirs(output_dir, exist_ok=True)

    pipeline_config = PipelineConfig(
        data=data_config,
        training=training_config,
        model_builder=model_builder_config,
        base_dir=output_dir,
    )

    # --- Comprehensive Final Configuration Logging ---
    SCRIPT_LOGGER.info("--- Final Configuration for Version 10 Run ---")
    SCRIPT_LOGGER.info(f"  Mixed Precision Enabled: {not cli_args.disable_mixed_precision}")
    SCRIPT_LOGGER.info(f"  Data file path: {pipeline_config.data.file_path}")
    if not os.path.exists(pipeline_config.data.file_path):
        SCRIPT_LOGGER.error(
            f"CRITICAL: Data file not found: {pipeline_config.data.file_path}. Exiting."
        )
        sys.exit(1)
    SCRIPT_LOGGER.info(f"  Run output base directory: {output_dir}")
    SCRIPT_LOGGER.info(
        "  Training Parameters: Epochs=%s, Batch Size=%s",
        pipeline_config.training.epochs,
        pipeline_config.training.batch_size,
    )
    SCRIPT_LOGGER.info(
        "  Data Processing: Time Steps=%s, Train Ratio=%s, Val Ratio=%s, Test Ratio=%s",
        pipeline_config.data.time_steps,
        pipeline_config.data.train_ratio,
        pipeline_config.data.val_ratio,
        pipeline_config.data.test_ratio,
    )
    SCRIPT_LOGGER.info(
        "  ModelBuilder V10 - Block Configurations: %s",
        json.dumps(pipeline_config.model_builder.block_configs, indent=4),
    )
    SCRIPT_LOGGER.info(f"  ModelBuilder V10 - Other Architectural Hyperparameters:")
    model_builder_dict = asdict(pipeline_config.model_builder)
    model_builder_dict.pop('block_configs', None)
    for key, value in model_builder_dict.items():
        SCRIPT_LOGGER.info(f"    {key}: {value}")

    # --- Execute Pipeline ---
    try:
        pipeline = TimeSeriesModel(pipeline_config)
        pipeline.run()
        SCRIPT_LOGGER.info("🎉 Version 10 pipeline completed successfully.")
    except Exception as e:
        SCRIPT_LOGGER.critical(f"❌ Version 10 pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    args = parse_arguments()
    setup_environment(enable_mixed_precision=(not args.disable_mixed_precision))
    main(args)