import tensorflow as tf
import json
import os
from sklearn.metrics import r2_score # mean_absolute_error, mean_squared_error are in Evaluator
import numpy as np
# from Evaluator import Evaluator # Evaluator is used in SavePlotsPerEpochCallback, ensure it's available
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback, EarlyStopping
import time 
from typing import Optional, Any, Dict, List 

# It's generally better to get the logger for the specific module
logger = logging.getLogger(__name__)
# Example: Set a default logging level or add a NullHandler
# logger.setLevel(logging.INFO)
# logger.addHandler(logging.NullHandler())

# Disable verbose logging from specific libraries if needed, typically done in the main script
# logging.getLogger('PIL').setLevel(logging.INFO)
# logging.getLogger('matplotlib').setLevel(logging.WARNING)
# logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)



class EpochTimerCallback(Callback):
    """
    A Keras Callback to precisely record the duration of each training epoch.

    This callback measures the time taken for each epoch and stores these durations.
    It also logs the duration at the end of each epoch and makes the list of
    durations accessible after training.

    Attributes:
        epoch_start_time (float): Timestamp taken at the beginning of an epoch.
        epoch_durations (List[float]): A list storing the duration (in seconds) of each completed epoch.
    """
    def __init__(self):
        super().__init__()
        self.epoch_start_time: float = 0.0
        self.epoch_durations: List[float] = [] # Stores duration of each epoch

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of an epoch."""
        self.epoch_start_time = time.time()
        logger.debug(f"Epoch {epoch + 1}: Start time recorded.")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of an epoch."""
        epoch_end_time = time.time()
        duration = epoch_end_time - self.epoch_start_time
        self.epoch_durations.append(duration)
        if logs is not None:
            logs['epoch_duration_sec'] = duration # Add to Keras logs for history object
        logger.info(f"Epoch {epoch + 1}: Duration = {duration:.2f} seconds.")

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of training."""
        self.epoch_durations = [] # Reset for a new training run
        logger.info("EpochTimerCallback: Training started, epoch durations list reset.")

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training."""
        logger.info(f"EpochTimerCallback: Training finished. Recorded {len(self.epoch_durations)} epoch durations.")
        # The list self.epoch_durations can be accessed by the Trainer or MainClass


class Trainer:
    """
    Manages the training process for a Keras model.

    This class takes a compiled Keras model, training and validation data,
    and training parameters (epochs, batch size). It fits the model using
    several callbacks, including early stopping, learning rate reduction,
    and custom callbacks for saving history, R2 scores, models per epoch,
    and plots per epoch.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 epochs: int = 1,
                 batch_size: int = 16,
                 history_path: Optional[str] = None,
                 main_model_instance: Optional[Any] = None,
                 early_stopping_patience: int = 20,     
                 reduce_lr_patience: int = 1,   
                 reduce_lr_factor: float = 0.1,  
                 min_lr: float = 5e-7            
                 ): # 'Any' can be replaced with the actual type of MainClass
        """
        Initializes the Trainer.

        Args:
            model (tf.keras.Model): The compiled Keras model to be trained.
            X_train (np.ndarray): Training feature data.
            y_train (np.ndarray): Training target data.
            X_val (np.ndarray): Validation feature data.
            y_val (np.ndarray): Validation target data.
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.
            history_path (Optional[str]): Path to save the training history (including R2 scores).
                                          Used by R2HistoryCallback and SaveHistoryCallback.
            main_model_instance (Optional[Any]): An instance of the main model-managing class
                                                 (e.g., TimeSeriesModel from MainClass.py).
                                                 Used by callbacks like SaveModelPerEpochCallback and
                                                 SavePlotsPerEpochCallback to access methods for
                                                 saving models, plots, and accessing scalers.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.epochs = epochs
        self.batch_size = batch_size
        self.history: Optional[tf.keras.callbacks.History] = None
        self.history_path = history_path
        self.main_model_instance = main_model_instance # Instance of MainClass (or similar)

                # Save hayperparameters callback
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.min_lr = min_lr

        # make sample of EpochTimerCallback
        self.epoch_timer_callback = EpochTimerCallback()

        logger.debug("Trainer initialized with model: %s", model.name if hasattr(model, 'name') else model)

    def train(self) -> Optional[tf.keras.callbacks.History]:
        """
        Executes the model training process using the configured Keras model and callbacks.

        This method assembles a list of callbacks, including standard Keras callbacks
        like `EarlyStopping` and `ReduceLROnPlateau`, as well as custom callbacks
        defined in this module (`EpochTimerCallback`, `R2HistoryCallback`, `SaveHistoryCallback`,
        `SaveModelPerEpochCallback`). It then calls `model.fit()` to start training.

        Returns:
            Optional[tf.keras.callbacks.History]: The Keras History object containing training
                                                 metrics and logs, or None if training failed.
                                                 The `epoch_durations_list` attribute of `main_model_instance`
                                                 (if provided to Trainer) will also be populated by
                                                 the `EpochTimerCallback`.
        """
        # Initialize an empty list to hold all callbacks for the training process.
        callbacks_list: List[Callback] = []

        # Step 1: Configure and add EarlyStopping callback.
        # This callback stops training when a monitored metric has stopped improving.
        # It requires validation data (X_val, y_val) to monitor 'val_loss'.
        if self.X_val is not None and self.y_val is not None:
            early_stopping = EarlyStopping(
                monitor='val_loss',                     # Metric to be monitored.
                patience=self.early_stopping_patience,  # Use instance attribute for patience.
                restore_best_weights=True,              # Restores model weights from the epoch with the best 'val_loss'.
                verbose=1                               # Logs when training stops early.
            )
            callbacks_list.append(early_stopping)
            logger.info(f"EarlyStopping callback enabled: monitor='val_loss', patience={self.early_stopping_patience}, restore_best_weights=True.")
        else:
            logger.warning("Validation data (X_val or y_val) not provided; EarlyStopping callback will not be used. "
                           "Training will run for the full number of configured epochs unless interrupted manually.")

        # Step 2: Configure and add ReduceLROnPlateau callback.
        # This callback reduces the learning rate when 'val_loss' has stopped improving.
        # It also benefits from having validation data.
        if self.X_val is not None and self.y_val is not None:
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.reduce_lr_factor,      # Use instance attribute for reduction factor.
                patience=self.reduce_lr_patience, # Use instance attribute for patience.
                min_lr=self.min_lr,                # Use instance attribute for minimum learning rate.
                verbose=1                          # Logs when the learning rate is reduced.
            )
            callbacks_list.append(reduce_lr)
            logger.info(f"ReduceLROnPlateau callback enabled: monitor='val_loss', factor={self.reduce_lr_factor}, patience={self.reduce_lr_patience}, min_lr={self.min_lr}.")
        else:
            logger.warning("Validation data (X_val or y_val) not provided; ReduceLROnPlateau callback will not be used, "
                           "or it will monitor 'loss' if validation_data is not passed to model.fit().")
        
        # Step 3: Add the EpochTimerCallback to the list of callbacks.
        # self.epoch_timer_callback should have been instantiated in the Trainer's __init__ method.
        callbacks_list.append(self.epoch_timer_callback)
        logger.info("EpochTimerCallback enabled for precise epoch duration tracking.")

        # Step 4: Add other existing callbacks (e.g., for saving history, R2 scores).
        # These are assumed to be defined as inner classes or imported.
        if self.history_path:
            # SaveHistoryCallback saves the Keras history object at the end of training.
            callbacks_list.append(self.SaveHistoryCallback(self.history_path))
            # R2HistoryCallback calculates R² on validation data per epoch.
            if self.X_val is not None and self.y_val is not None:
                 callbacks_list.append(self.R2HistoryCallback(self.history_path, self.X_val, self.y_val))
                 logger.info(f"R2HistoryCallback enabled. History saving callbacks configured for path: {self.history_path}")
            else:
                 logger.warning(f"R2HistoryCallback not added as validation data is missing. SaveHistoryCallback still configured for: {self.history_path}")
        else:
            logger.warning("`history_path` not provided. SaveHistoryCallback and R2HistoryCallback will not be used.")

        # Add SaveModelPerEpochCallback if main_model_instance is provided and has the required method.
        if self.main_model_instance:
            if hasattr(self.main_model_instance, 'save_model_per_epoch'):
                 callbacks_list.append(self.SaveModelPerEpochCallback(self.main_model_instance))
                 logger.info("SaveModelPerEpochCallback enabled (delegates to main_model_instance.save_model_per_epoch).")
            # The SavePlotsPerEpochCallback is assumed to be managed elsewhere or disabled by default
            # due to its potential resource intensity and complexity.
            logger.debug("SavePlotsPerEpochCallback is currently managed outside this explicit callback list or disabled by default in the Trainer's `train` method.")

        logger.info(f"Starting model training with {len(callbacks_list)} callbacks. "
                    f"Target epochs: {self.epochs}, Batch size: {self.batch_size}.")
        
        # Prepare validation_data argument for model.fit()
        validation_data_for_fit = (self.X_val, self.y_val) if self.X_val is not None and self.y_val is not None else None
            
        try:
            # Execute the training process.
            self.history = self.model.fit(
                self.X_train, self.y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=validation_data_for_fit,
                verbose=1, # Standard progress bar logging for Keras.
                callbacks=callbacks_list # Pass the comprehensive list of callbacks.
            )
            logger.info("Model training process finished.")

            if self.history is None or not hasattr(self.history, 'history') or not self.history.history:
                logger.warning("Keras `model.history` object is None or empty after training. "
                               "This might occur if training was interrupted very early or an error occurred.")
            else:
                logger.debug(f"Training history recorded. Available metric keys: {list(self.history.history.keys())}")
            
            # Step 5: Transfer the list of epoch durations to the MainClass instance.
            # This makes the detailed timing information available for saving and analysis in MainClass.
            # self.epoch_timer_callback.epoch_durations contains the list of durations.
            if self.main_model_instance and hasattr(self.epoch_timer_callback, 'epoch_durations'):
                # The `main_model_instance` (e.g., an instance of TimeSeriesModel)
                # should have an attribute like `epoch_durations_list` initialized (e.g., as an empty list).
                self.main_model_instance.epoch_durations_list = self.epoch_timer_callback.epoch_durations
                logger.info(f"Epoch durations list (length: {len(self.main_model_instance.epoch_durations_list)}) "
                            "has been passed to the main_model_instance.")

            return self.history

        except Exception as e:
            logger.error(f"A critical error occurred during `model.fit()`: {e}", exc_info=True)
            return None # Indicate that training failed.
    class R2HistoryCallback(Callback): # Ensure this class is defined within Trainer or imported
        def __init__(self, history_path: str, X_val: np.ndarray, y_val: np.ndarray):
            super().__init__()
            self.history_path = history_path
            self.X_val = X_val
            self.y_val = y_val
            self.epoch_val_r2_scores: List[float] = []
            logger.debug("R2HistoryCallback initialized.")

        def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
            self.epoch_val_r2_scores = [] 
            logger.debug("R2HistoryCallback: R² scores list reset at train begin.")

        def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
            logs = logs if logs is not None else {}
            if self.X_val is None or self.y_val is None or self.X_val.shape[0] == 0:
                logger.warning(f"R2HistoryCallback: Validation data is missing or empty for epoch {epoch + 1}. Skipping R² calculation.")
                return
            try:
                y_pred_val_scaled = self.model.predict(self.X_val, verbose=0)
                current_y_val = self.y_val.reshape(-1, 1) if self.y_val.ndim == 1 else self.y_val
                current_y_pred_val = y_pred_val_scaled.reshape(-1, 1) if y_pred_val_scaled.ndim == 1 else y_pred_val_scaled
                if current_y_val.shape[0] != current_y_pred_val.shape[0]:
                    logger.error(f"R2HistoryCallback: Shape mismatch for R² calculation at epoch {epoch + 1}.")
                    return
                r2 = r2_score(current_y_val, current_y_pred_val)
                self.epoch_val_r2_scores.append(float(r2))
                logs['val_r2_custom'] = float(r2) # Add to Keras logs
                logger.info(f"Epoch {epoch + 1}: Custom Validation R² = {r2:.6f}")
            except Exception as e:
                logger.error(f"Error in R2HistoryCallback for epoch {epoch + 1}: {e}", exc_info=True)


    class SaveHistoryCallback(Callback): # Ensure this class is defined within Trainer or imported
        def __init__(self, history_path: str):
            super().__init__()
            self.history_path = history_path
            logger.debug(f"SaveHistoryCallback initialized, will save to: {history_path}")

        def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
            if not (hasattr(self.model, 'history') and self.model.history is not None and hasattr(self.model.history, 'history')):
                logger.warning("SaveHistoryCallback: Model history not found. Cannot save.")
                return
            history_to_save = {k: [float(vi) for vi in v] if isinstance(v, list) else float(v) 
                               for k, v in self.model.history.history.items()}
            try:
                history_dir = os.path.dirname(self.history_path)
                if history_dir: os.makedirs(history_dir, exist_ok=True)
                with open(self.history_path, 'w') as f:
                    json.dump(history_to_save, f, indent=4)
                logger.info(f"SaveHistoryCallback: Full training history saved to: {self.history_path}")
            except Exception as e:
                logger.error(f"Error saving history with SaveHistoryCallback to {self.history_path}: {e}", exc_info=True)


    class SaveModelPerEpochCallback(Callback): # Ensure this class is defined within Trainer or imported
        def __init__(self, main_model_instance: Any):
            super().__init__()
            self.main_model_instance = main_model_instance
            if not (self.main_model_instance and hasattr(self.main_model_instance, 'save_model_per_epoch')):
                logger.warning("SaveModelPerEpochCallback: main_model_instance invalid or lacks 'save_model_per_epoch'.")
                self.main_model_instance = None 
            else:
                logger.debug("SaveModelPerEpochCallback initialized.")

        def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
            if self.main_model_instance:
                try:
                    self.main_model_instance.save_model_per_epoch(epoch + 1)
                except Exception as e:
                    logger.error(f"Error in SaveModelPerEpochCallback (epoch {epoch + 1}): {e}", exc_info=True)

    class SavePlotsPerEpochCallback(Callback):
        """
        Keras Callback to generate and save various evaluation plots at the end of each epoch.
        Relies on `main_model_instance` for configuration (run_dir, scaler_y) and potentially
        an Evaluator instance or similar plotting capabilities.

        Note: This callback can be resource-intensive if plotting involves significant computation
        or I/O at every epoch. Consider its necessity or frequency.
        """
        def __init__(self, main_model_instance: Any, X_val: np.ndarray, y_val: np.ndarray):
            """
            Initializes the SavePlotsPerEpochCallback.

            Args:
                main_model_instance (Any): An instance of the main model-managing class.
                                           Expected to have `run_dir` and `data_loader.scaler_y`.
                X_val (np.ndarray): Validation feature data for generating predictions.
                y_val (np.ndarray): Validation target data for comparison.
            """
            super().__init__()
            self.main_model_instance = main_model_instance
            self.X_val = X_val
            self.y_val = y_val
            self.train_losses: List[float] = []
            self.val_losses: List[float] = []
            logger.debug("SavePlotsPerEpochCallback initialized.")

        def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
            """
            Generates and saves loss plots, prediction plots, and error distribution plots.
            """
            logs = logs or {}
            current_epoch_num = epoch + 1
            try:
                # Essential checks for main_model_instance and its attributes
                if self.main_model_instance is None:
                    logger.error("Main model instance is None in SavePlotsPerEpochCallback.")
                    return
                if not hasattr(self.main_model_instance, 'run_dir') or \
                   not hasattr(self.main_model_instance, 'data_loader') or \
                   not hasattr(self.main_model_instance.data_loader, 'scaler_y'):
                    logger.error("main_model_instance is missing required attributes (run_dir, data_loader.scaler_y).")
                    return

                # Check for validation data
                if self.X_val is None or self.y_val is None or self.X_val.shape[0] == 0:
                    logger.warning("Validation data (X_val or y_val) is None or empty in SavePlotsPerEpochCallback. Skipping plots.")
                    return

                train_loss = logs.get('loss')
                val_loss = logs.get('val_loss')

                if train_loss is None or val_loss is None:
                    logger.warning("Epoch %d: 'loss' or 'val_loss' is missing in logs. Skipping plot generation.", current_epoch_num)
                    return

                self.train_losses.append(float(train_loss))
                self.val_losses.append(float(val_loss))

                # Create a directory for this epoch's plots
                # This can create many directories. Consider if a single plot directory updated is better.
                epoch_plots_dir = os.path.join(self.main_model_instance.run_dir, 'epoch_plots', f'epoch_{current_epoch_num:03d}')
                os.makedirs(epoch_plots_dir, exist_ok=True)

                # --- 1. Save Loss Plot (cumulative up to current epoch) ---
                try:
                    plt.figure(figsize=(10, 6))
                    plt.plot(self.train_losses, label='Train Loss', color='blue')
                    plt.plot(self.val_losses, label='Validation Loss', color='orange')
                    plt.title(f'Model Loss up to Epoch {current_epoch_num}')
                    plt.ylabel('Loss')
                    plt.xlabel('Epoch')
                    plt.legend(loc='upper right')
                    plt.grid(True)
                    loss_plot_path = os.path.join(epoch_plots_dir, 'loss_plot.png')
                    plt.savefig(loss_plot_path)
                    plt.close()
                    logger.info("Epoch %d: Cumulative loss plot saved to %s", current_epoch_num, loss_plot_path)
                except Exception as e:
                    logger.error(f"Epoch {current_epoch_num}: Error saving loss plot: {e}", exc_info=True)

                # --- Predictions and other plots (can be intensive per epoch) ---
                # Skip for the first epoch if predictions might be unstable or not meaningful
                if epoch == 0 and False: # Set to True to skip first epoch plots
                    logger.info("Skipping detailed plots on the first epoch.")
                    return

                try:
                    y_pred_val_scaled = self.model.predict(self.X_val, verbose=0)
                    if y_pred_val_scaled is None:
                        logger.error("Epoch %d: Predictions on validation data are None.", current_epoch_num)
                        return

                    # Rescale predictions and actual values
                    scaler_y = self.main_model_instance.data_loader.scaler_y
                    y_pred_val_rescaled = scaler_y.inverse_transform(y_pred_val_scaled)
                    y_val_rescaled = scaler_y.inverse_transform(self.y_val)

                    # At this point, you would typically use an Evaluator instance or similar
                    # methods from main_model_instance to generate other plots.
                    # For this example, let's just plot predictions vs actual.
                    # A full Evaluator integration would be more robust.

                    # --- 2. Save Prediction Plot ---
                    plt.figure(figsize=(12, 6))
                    plt.plot(y_val_rescaled, label='Actual Validation Data', color='blue', alpha=0.7)
                    plt.plot(y_pred_val_rescaled, label='Predicted Validation Data', color='red', linestyle='--')
                    plt.title(f'Validation Predictions vs Actual - Epoch {current_epoch_num}')
                    plt.xlabel('Sample Index')
                    plt.ylabel('Value')
                    plt.legend()
                    plt.grid(True)
                    prediction_plot_path = os.path.join(epoch_plots_dir, 'prediction_plot.png')
                    plt.savefig(prediction_plot_path)
                    plt.close()
                    logger.info("Epoch %d: Prediction plot saved to %s", current_epoch_num, prediction_plot_path)

                    # --- 3. Save Error Distribution Plot ---
                    errors_val = y_val_rescaled - y_pred_val_rescaled
                    plt.figure(figsize=(10, 5))
                    plt.hist(errors_val, bins=50, color='purple', edgecolor='black', alpha=0.7)
                    plt.title(f'Distribution of Validation Prediction Errors - Epoch {current_epoch_num}')
                    plt.xlabel('Prediction Error (Actual - Predicted)')
                    plt.ylabel('Frequency')
                    plt.grid(True)
                    error_dist_plot_path = os.path.join(epoch_plots_dir, 'error_distribution_plot.png')
                    plt.savefig(error_dist_plot_path)
                    plt.close()
                    logger.info("Epoch %d: Error distribution plot saved to %s", current_epoch_num, error_dist_plot_path)

                except Exception as e:
                    logger.error(f"Epoch {current_epoch_num}: Error during prediction or plotting of detailed metrics: {e}", exc_info=True)

                logger.info("Epoch %d: All plots saved in %s.", current_epoch_num, epoch_plots_dir)

            except Exception as e:
                logger.error(f"Epoch {current_epoch_num}: An unexpected error occurred in SavePlotsPerEpochCallback: {e}", exc_info=True)