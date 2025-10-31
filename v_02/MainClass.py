# MainClass.py
# This module defines the TimeSeriesModel class, which orchestrates the entire
# time series forecasting pipeline, from data loading to model evaluation and saving.
# It is designed to be configurable and reusable across different model versions.

import os
import datetime
import json
import logging
import time # For precise timing of operations
import numpy as np # For numerical operations like argmin
import tensorflow as tf # For TensorFlow/Keras specific types and operations
from typing import Optional, List, Dict, Any, Tuple # For type hinting

# Import other necessary modules from the project structure
# These are assumed to be in the same directory or correctly configured in PYTHONPATH.
from ModelBuilder import ModelBuilder # The specific ModelBuilder (e.g., V1, V2, etc.) will be used
from Trainer import Trainer         # Handles the training loop and callbacks
from DataLoader import DataLoader     # Handles data loading and preprocessing
from Evaluator import Evaluator       # Handles model evaluation and plotting
from HistoryManager import HistoryManager # Manages saving/loading of training history

# Module-level logger. The handlers (e.g., StreamHandler, FileHandler for the run)
# are typically configured in Run.py (for console) and within this class (for run-specific file).
logger = logging.getLogger(__name__)

class TimeSeriesModel:
    """
    Manages the complete lifecycle of a time series forecasting model.

    This class orchestrates:
    1.  Setup of a unique run directory for storing all outputs.
    2.  Configuration of a run-specific file logger.
    3.  Data loading and preprocessing via DataLoader.
    4.  Neural network model construction via ModelBuilder.
    5.  Model training via Trainer, which includes callbacks for:
        - Early stopping to prevent overfitting.
        - Adaptive learning rate reduction.
        - Precise epoch timing (EpochTimerCallback).
        - Custom R2 metric calculation per epoch (R2HistoryCallback).
        - Saving full training history (SaveHistoryCallback).
        - Saving model checkpoints per epoch (delegated via SaveModelPerEpochCallback).
    6.  Saving final training history and a summary of training (including timing)
        to the hyperparameters file.
    7.  Model evaluation via Evaluator (predictions, metrics, plots).
    8.  Saving the final trained model in multiple standard formats.
    """

    def __init__(self,
                 # Core pipeline parameters
                 file_path: str,
                 base_dir: str = "TimeSeries_Project_Runs/",
                 # Data loading and splitting parameters
                 time_steps: int = 60, # Default based on common practice, override in Run.py
                 train_ratio: float = 0.94,
                 val_ratio: float = 0.03,
                 test_ratio: float = 0.03,
                 # Model training parameters
                 epochs: int = 20, # Default, override in Run.py
                 batch_size: int = 1120, # Default, override in Run.py
                 # Model architecture parameters (passed to ModelBuilder)
                 block_configs: Optional[List[Dict[str, Any]]] = None, # For ModelBuilders that use it
                 model_builder_params: Optional[Dict[str, Any]] = None,
                 # Trainer callback configurations (passed to Trainer)
                 early_stopping_patience: int = 60,
                 reduce_lr_patience: int = 1,
                 reduce_lr_factor: float = 0.1, # Default for ReduceLROnPlateau
                 min_lr: float = 5e-7         # Default for ReduceLROnPlateau
                 ):
        """
        Initializes the TimeSeriesModel pipeline orchestrator.

        Args:
            file_path (str): Path to the CSV data file.
            base_dir (str): Base directory to store all outputs for this and other runs.
                            A timestamped subdirectory will be created here for the current run.
            time_steps (int): Number of past time steps for input sequences (passed to DataLoader).
            train_ratio (float): Proportion of data for the training set (passed to DataLoader).
            val_ratio (float): Proportion of data for the validation set (passed to DataLoader).
            test_ratio (float): Proportion of data for the test set (passed to DataLoader).
            epochs (int): Number of training epochs (passed to Trainer).
            batch_size (int): Batch size for training (passed to Trainer).
            block_configs (Optional[List[Dict[str, Any]]]): Configuration for convolutional blocks,
                passed to ModelBuilder. Structure depends on the specific ModelBuilder version.
            model_builder_params (Optional[Dict[str, Any]]): Dictionary of other parameters
                to be passed to the ModelBuilder constructor.
            early_stopping_patience (int): Patience for the EarlyStopping callback (passed to Trainer).
            reduce_lr_patience (int): Patience for the ReduceLROnPlateau callback (passed to Trainer).
            reduce_lr_factor (float): Factor by which learning rate is reduced (passed to Trainer).
            min_lr (float): Minimum learning rate for ReduceLROnPlateau (passed to Trainer).
        """
        self.file_path = file_path
        self.base_dir = base_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.block_configs = block_configs if block_configs is not None else []
        self.model_builder_params = model_builder_params if model_builder_params is not None else {}

        # Store data and callback parameters
        self.time_steps = time_steps
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.min_lr = min_lr

        # Initialize attributes that will be populated during the pipeline
        self.epoch_durations_list: List[float] = [] # To be populated by Trainer's EpochTimerCallback
        self.training_summary_for_hyperparameters: Dict[str, Any] = {} # For storing timing and best epoch info

        # --- Setup Run-Specific Directory and File Logger ---
        self._setup_run_directory_and_logging()

        # --- Define Paths for Artifacts ---
        self._define_artifact_paths()

        # --- Instantiate Helper Classes ---
        self.data_loader = DataLoader(
            file_path=self.file_path, time_steps=self.time_steps,
            train_ratio=self.train_ratio, val_ratio=self.val_ratio, test_ratio=self.test_ratio
        )
        self.model: Optional[tf.keras.Model] = None # Will be built by ModelBuilder
        self.trainer: Optional[Trainer] = None       # Will be instantiated in run()
        self.evaluator: Optional[Evaluator] = None   # Will be instantiated in run()
        self.history_manager = HistoryManager(self.history_path)

        logger.info(f"TimeSeriesModel initialized. All outputs for this run will be in: {self.run_dir}")
        logger.info(f"Detailed log file for this run: {self.log_file_path}")

    def _setup_run_directory_and_logging(self):
        """Creates a unique timestamped directory for the current run and sets up file logging."""
        os.makedirs(self.base_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") # Added microseconds for more uniqueness
        self.run_dir = os.path.join(self.base_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        logger.info(f"Run directory created: {self.run_dir}")

        self.log_file_path = os.path.join(self.run_dir, 'run_pipeline_log.txt')
        # Create a file handler specific to this run
        self.file_log_handler = logging.FileHandler(self.log_file_path, mode='w') # 'w' to overwrite if run again (unlikely with timestamp)
        self.file_log_handler.setLevel(logging.DEBUG) # Capture detailed logs in the file
        log_formatter = logging.Formatter('%(asctime)s [%(levelname)-8s] %(name)s: %(message)s (%(filename)s:%(lineno)d)')
        self.file_log_handler.setFormatter(log_formatter)
        
        # Add this handler to the root logger to capture logs from all modules for this run
        logging.getLogger().addHandler(self.file_log_handler)

    def _define_artifact_paths(self):
        """Defines paths for saving various artifacts within the run directory."""
        self.history_path = os.path.join(self.run_dir, 'training_history.json')
        self.hyperparameters_path = os.path.join(self.run_dir, 'hyperparameters_and_summary.json')
        self.model_keras_path = os.path.join(self.run_dir, 'model_final.keras')
        self.model_h5_path = os.path.join(self.run_dir, 'model_final.h5')
        self.saved_model_dir_path = os.path.join(self.run_dir, 'model_final_tf_savedmodel')
        # Plot paths are typically managed by the Evaluator within its own output structure or run_dir.

    def save_hyperparameters(self):
        """
        Saves key hyperparameters and the training summary for this run to a JSON file.
        This method can be called multiple times; it will overwrite the file with the latest info.
        """
        hyperparams_to_save = {
            'run_info': {
                'timestamp': self.run_dir.split('_')[-2], # Adjusted for _%f
                'run_directory': self.run_dir,
                'log_file': self.log_file_path,
            },
            'data_parameters': {
                'file_path': self.file_path,
                'time_steps': self.data_loader.time_steps if hasattr(self, 'data_loader') and self.data_loader else self.time_steps,
                'train_ratio': self.data_loader.train_ratio if hasattr(self, 'data_loader') and self.data_loader else self.train_ratio,
                'val_ratio': self.data_loader.val_ratio if hasattr(self, 'data_loader') and self.data_loader else self.val_ratio,
                'test_ratio': self.data_loader.test_ratio if hasattr(self, 'data_loader') and self.data_loader else self.test_ratio,
            },
            'training_parameters': {
                'epochs_configured': self.epochs,
                'batch_size': self.batch_size,
                'early_stopping_patience': self.early_stopping_patience,
                'reduce_lr_patience': self.reduce_lr_patience,
                'reduce_lr_factor': self.reduce_lr_factor,
                'min_lr': self.min_lr,
            },
            'model_architecture_parameters': {
                'block_configs': self.block_configs, # For ModelBuilders that use this list
                **self.model_builder_params      # For ModelBuilders that take individual params
            }
        }
        
        # Add the training_summary dictionary if it has been populated (after training)
        if self.training_summary_for_hyperparameters: # Check if dict is not empty
            hyperparams_to_save['training_summary'] = self.training_summary_for_hyperparameters
        
        try:
            with open(self.hyperparameters_path, 'w') as f:
                json.dump(hyperparams_to_save, f, indent=4, default=str) # default=str for non-serializable types
            logger.info(f"⚙️ Hyperparameters and training summary saved to: {self.hyperparameters_path}")
        except Exception as e:
            logger.error(f"❌ Error saving hyperparameters to {self.hyperparameters_path}: {e}", exc_info=True)

    def save_model_all_formats(self):
        """Saves the trained model in Keras native, H5, and TensorFlow SavedModel formats."""
        if self.model is None:
            logger.error("Model is not available (None). Cannot save model.")
            return

        logger.info("💾 Saving the final trained model in all standard formats...")
        try:
            self.model.save(self.model_keras_path)
            logger.info(f"✅ Model saved successfully in Keras native format to: {self.model_keras_path}")

            self.model.save(self.model_h5_path) # Keras will issue a warning about legacy format
            logger.info(f"✅ Model saved successfully in H5 format to: {self.model_h5_path}")

            self.model.export(self.saved_model_dir_path)
            logger.info(f"✅ Model exported successfully in TensorFlow SavedModel format to: {self.saved_model_dir_path}")
        except Exception as e:
            logger.error(f"❌ An error occurred during model saving/exporting: {e}", exc_info=True)

    def save_model_per_epoch(self, epoch_num: int):
        """
        Saves the model at the end of a specified epoch. Intended for use by a Keras callback.

        Args:
            epoch_num (int): The epoch number (typically 1-indexed from the callback).
        """
        if self.model is None:
            logger.error(f"Model is not available (None). Cannot save model for epoch {epoch_num}.")
            return

        epoch_models_dir = os.path.join(self.run_dir, "epoch_models")
        os.makedirs(epoch_models_dir, exist_ok=True)
        epoch_model_path = os.path.join(epoch_models_dir, f'model_epoch_{epoch_num:03d}.keras')

        try:
            self.model.save(epoch_model_path)
            logger.info(f"💾 Checkpoint: Model (Keras format) saved for epoch {epoch_num} to: {epoch_model_path}")
        except Exception as e:
            logger.error(f"❌ Error saving model checkpoint for epoch {epoch_num}: {e}", exc_info=True)

    def run(self):
        """
        Executes the complete model training and evaluation pipeline.
        """
        logger.info(f"🚀 Starting the model training and evaluation pipeline for run: {self.run_dir}")
        run_successful = False
        pipeline_start_time = time.time()

        try:
            # --- 1. Data Loading & Preprocessing ---
            logger.info("Step 1: Loading and preprocessing data...")
            data_load_start_time = time.time()
            X_train, X_val, X_test, y_train, y_val, y_test, scaler_y = self.data_loader.get_data()
            data_load_end_time = time.time()
            logger.info(f"Data loading and preprocessing completed in {data_load_end_time - data_load_start_time:.3f} seconds.")

            if X_train is None or X_train.shape[0] == 0:
                logger.critical("CRITICAL: Training data (X_train) is empty. Aborting pipeline.")
                self._cleanup_logging() # Clean up logger before exiting
                return
            
            logger.info(f"Data shapes: X_train: {X_train.shape}, y_train: {y_train.shape}" +
                        (f", X_val: {X_val.shape}, y_val: {y_val.shape}" if X_val is not None and X_val.shape[0] > 0 else ", No validation data") +
                        (f", X_test: {X_test.shape}, y_test: {y_test.shape}" if X_test is not None and X_test.shape[0] > 0 else ", No test data"))

            # --- 2. Save initial hyperparameters (training summary will be added later) ---
            self.save_hyperparameters()

            # --- 3. Model Building ---
            logger.info("Step 2: Building the model...")
            model_build_start_time = time.time()
            model_builder = ModelBuilder(
                time_steps=X_train.shape[1],
                num_features=X_train.shape[2],
                block_configs=self.block_configs, # Pass block_configs (may be empty or used by specific ModelBuilder)
                **self.model_builder_params    # Pass all other model-specific params
            )
            self.model = model_builder.build_model()
            model_build_end_time = time.time()
            logger.info(f"✅ Model built and compiled successfully in {model_build_end_time - model_build_start_time:.3f} seconds.")
            model_summary_lines = []
            self.model.summary(print_fn=lambda x: model_summary_lines.append(x), line_length=120)
            logger.info(f"Model Summary:\n" + "\n".join(model_summary_lines))

            # --- 4. Model Training ---
            logger.info(f"Step 3: Training the model for {self.epochs} configured epochs with batch size {self.batch_size}...")
            self.trainer = Trainer(
                model=self.model,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                epochs=self.epochs, batch_size=self.batch_size,
                history_path=self.history_path,
                main_model_instance=self, # Crucial for callbacks to access this instance
                early_stopping_patience=self.early_stopping_patience,
                reduce_lr_patience=self.reduce_lr_patience,
                reduce_lr_factor=self.reduce_lr_factor,
                min_lr=self.min_lr
            )
            training_history: Optional[tf.keras.callbacks.History] = self.trainer.train()
            
            actual_epochs_run = len(self.epoch_durations_list)
            total_training_time = sum(self.epoch_durations_list) if self.epoch_durations_list else 0.0
            logger.info(f"✅ Model training phase finished. Actual epochs run: {actual_epochs_run}. "
                        f"Total precise training time: {total_training_time:.3f} seconds.")

            # --- Process and Log Training Summary (including best epoch timing) ---
            best_epoch_num_val_loss = -1
            time_to_best_epoch_val_loss = -1.0
            best_val_loss_value = float('inf')

            if training_history and hasattr(training_history, 'history') and 'val_loss' in training_history.history:
                val_losses = training_history.history['val_loss']
                if val_losses:
                    best_epoch_idx_val_loss = np.argmin(val_losses)
                    best_epoch_num_val_loss = best_epoch_idx_val_loss + 1
                    best_val_loss_value = val_losses[best_epoch_idx_val_loss]
                    
                    if self.epoch_durations_list and actual_epochs_run >= best_epoch_num_val_loss:
                        time_to_best_epoch_val_loss = sum(self.epoch_durations_list[:best_epoch_num_val_loss])
                        logger.info(f"Best validation loss ({best_val_loss_value:.6f}) achieved at epoch {best_epoch_num_val_loss}.")
                        logger.info(f"Precise time to reach best validation loss epoch: {time_to_best_epoch_val_loss:.3f} seconds.")
                    else:
                        logger.warning("Epoch durations list not available/sufficient for precise time to best val_loss epoch.")
            else:
                logger.warning("'val_loss' not found in Keras history or history unavailable. Cannot determine best epoch.")
            
            self.training_summary_for_hyperparameters = {
                "total_training_time_seconds": round(total_training_time, 3),
                "actual_epochs_run": actual_epochs_run,
                "best_val_loss_epoch_num (1-based)": best_epoch_num_val_loss,
                "best_val_loss_value": round(best_val_loss_value, 6) if best_val_loss_value != float('inf') else "N/A",
                "time_to_best_val_loss_epoch_seconds": round(time_to_best_epoch_val_loss, 3) if time_to_best_epoch_val_loss > -1.0 else "N/A",
                "avg_time_per_epoch_seconds": round(total_training_time / actual_epochs_run, 3) if actual_epochs_run > 0 else "N/A",
                "epoch_durations_sec_list": [round(d, 3) for d in self.epoch_durations_list]
            }
            self.save_hyperparameters() # Re-save to include training summary

            # --- 5. Save Final Training History (JSON) ---
            if training_history and hasattr(training_history, 'history'):
                self.history_manager.save_history(training_history)
                # SaveHistoryCallback in Trainer also saves this, this is an explicit final save.
                logger.info(f"💾 Final Keras training history explicitly saved by HistoryManager to: {self.history_path}")

            # --- 6. Model Evaluation ---
            logger.info("Step 4: Evaluating the trained model...")
            evaluation_start_time = time.time()
            self.evaluator = Evaluator(
                model=self.model, X_test=X_test, y_test=y_test, X_val=X_val, y_val=y_val,
                scaler_y=scaler_y, run_dir=self.run_dir, history_manager=self.history_manager
            )
            self.evaluator.predict()
            self.evaluator.calculate_metrics()
            self.evaluator.save_metrics_to_file()
            
            if training_history and hasattr(training_history, 'history') and training_history.history:
                self.evaluator.plot_loss(training_history)
                self.evaluator.plot_metric_evolution(training_history, 'mae', 'mae_evolution_plot.png')
                self.evaluator.plot_metric_evolution(training_history, 'mse', 'mse_evolution_plot.png')
                # Plot custom R2 if it was added to history by R2HistoryCallback
                if 'val_r2_custom' in training_history.history:
                     self.evaluator.plot_metric_evolution(training_history, 'val_r2_custom', 'val_r2_custom_evolution_plot.png')
            
            if self.evaluator.r2_test is not None: self.evaluator.plot_r2_bar(self.evaluator.r2_test, dataset_name="Test")
            if self.evaluator.r2_val is not None: self.evaluator.plot_r2_bar(self.evaluator.r2_val, dataset_name="Validation")
            self.evaluator.plot_predictions()
            self.evaluator.plot_error_distribution()
            evaluation_end_time = time.time()
            logger.info(f"✅ Model evaluation completed in {evaluation_end_time - evaluation_start_time:.3f} seconds.")

            # --- 7. Final Model Saving ---
            self.save_model_all_formats()
            run_successful = True

        except Exception as e:
            logger.critical(f"❌ AN UNEXPECTED CRITICAL ERROR OCCURRED in the pipeline for run {self.run_dir}: {e}", exc_info=True)
        finally:
            pipeline_end_time = time.time()
            total_pipeline_time = pipeline_end_time - pipeline_start_time
            logger.info(f"Total pipeline execution time: {total_pipeline_time:.3f} seconds.")
            if run_successful:
                logger.info(f"🎉 Model training and evaluation pipeline completed successfully for run: {self.run_dir}")
            else:
                logger.error(f"🔥 Pipeline execution FAILED for run: {self.run_dir}. Check logs at '{self.log_file_path}' for details.")
            
            # Clean up the file handler added to the root logger by this instance
            self._cleanup_logging()

    def _cleanup_logging(self):
        """Closes and removes the run-specific file log handler from the root logger."""
        if hasattr(self, 'file_log_handler') and self.file_log_handler is not None:
            root_logger = logging.getLogger()
            if self.file_log_handler in root_logger.handlers:
                try:
                    self.file_log_handler.close()
                    root_logger.removeHandler(self.file_log_handler)
                    logger.debug(f"File log handler for {self.log_file_path} closed and removed from root logger.")
                except Exception as e:
                    logger.error(f"Error closing/removing file log handler: {e}", exc_info=True)
            self.file_log_handler = None # Prevent further attempts to close/remove
