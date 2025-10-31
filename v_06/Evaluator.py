import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from typing import Optional, Tuple, Any # Added Any for history object
import os
import logging
import tensorflow as tf # Added for type hinting Keras History

# It's good practice to get the logger for the specific module
logger = logging.getLogger(__name__)
# Example: Set a default logging level or add a NullHandler
# logger.setLevel(logging.INFO) # Changed from CRITICAL for more verbosity during development
# logger.addHandler(logging.NullHandler()) # For library use

# The following handler setup was in the original code.
# If logging is configured centrally in the application (e.g., MainClass or Run.py),
# this might not be necessary here or could lead to duplicate handlers/messages.
# Consider removing if logging is handled at a higher level.
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - Evaluator - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# if not logger.hasHandlers(): # Add handler only if no handlers are configured
#    logger.addHandler(handler)


class Evaluator:
    """
    Handles the evaluation of a trained Keras model.

    This class performs predictions on test and validation datasets,
    calculates various performance metrics (MAE, MSE, RMSE, R2),
    and generates plots for visualizing model performance, including
    loss curves, metric evolution, prediction comparisons, and error distributions.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        X_test: Optional[np.ndarray], # Made Optional
        y_test: Optional[np.ndarray], # Made Optional
        scaler_y: Any, # Should be a scikit-learn scaler, e.g., StandardScaler
        run_dir: str = "", # Directory to save plots and metrics
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        history_manager: Optional[Any] = None # Type hint for HistoryManager if available
    ):
        """
        Initializes the Evaluator.

        Args:
            model (tf.keras.Model): The trained Keras model to be evaluated.
            X_test (Optional[np.ndarray]): Test feature data. Can be None if only validation is evaluated.
            y_test (Optional[np.ndarray]): Test target data. Can be None.
            scaler_y (Any): The fitted scaler object (e.g., StandardScaler from scikit-learn)
                            used for the target variable. This is crucial for inverse transforming
                            predictions and actual values back to their original scale.
            run_dir (str): The directory where evaluation outputs (plots, metrics files) will be saved.
            X_val (Optional[np.ndarray]): Validation feature data.
            y_val (Optional[np.ndarray]): Validation target data.
            history_manager (Optional[Any]): An instance of HistoryManager, if available,
                                             for accessing training history data. (Not directly used in current methods).
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.scaler_y = scaler_y
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True) # Ensure run_dir exists

        # Initialize attributes for predictions and rescaled values
        self.y_pred_test_scaled: Optional[np.ndarray] = None
        self.y_pred_val_scaled: Optional[np.ndarray] = None
        self.y_pred_test_rescaled: Optional[np.ndarray] = None
        self.y_pred_val_rescaled: Optional[np.ndarray] = None
        self.y_test_rescaled: Optional[np.ndarray] = None
        self.y_val_rescaled: Optional[np.ndarray] = None

        # Initialize attributes for metrics
        self.mae_test: Optional[float] = None
        self.mse_test: Optional[float] = None
        self.rmse_test: Optional[float] = None
        self.r2_test: Optional[float] = None
        self.mae_val: Optional[float] = None
        self.mse_val: Optional[float] = None
        self.rmse_val: Optional[float] = None
        self.r2_val: Optional[float] = None

        self.history_manager = history_manager # Currently not used actively in methods
        logger.debug("Evaluator initialized. Output directory: %s", self.run_dir)


    def predict(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Performs predictions on the test and/or validation datasets and rescales them.

        The predictions and actual target values are inverse-transformed using `self.scaler_y`
        to bring them back to their original data scale. Results are stored in instance attributes.

        Returns:
            Tuple containing (all can be None if corresponding data is not available):
                - y_pred_test_rescaled (np.ndarray): Rescaled predictions for the test set.
                - y_test_rescaled (np.ndarray): Rescaled actual values for the test set.
                - y_pred_val_rescaled (np.ndarray): Rescaled predictions for the validation set.
                - y_val_rescaled (np.ndarray): Rescaled actual values for the validation set.
        """
        logger.info("🔄 Starting prediction process for test and/or validation datasets...")

        # --- Test Set Predictions ---
        if self.X_test is not None and self.y_test is not None and self.X_test.shape[0] > 0:
            try:
                logger.info("⏳ Running predictions on the test dataset...")
                self.y_pred_test_scaled = self.model.predict(self.X_test, verbose=0)
                self.y_pred_test_rescaled = self.scaler_y.inverse_transform(self.y_pred_test_scaled)
                self.y_test_rescaled = self.scaler_y.inverse_transform(self.y_test) # y_test should be scaled form
                logger.info("✅ Test dataset prediction and rescaling completed.")
            except Exception as e:
                logger.error(f"❌ Error during test prediction or rescaling: {e}", exc_info=True)
                self.y_pred_test_rescaled, self.y_test_rescaled = None, None # Reset on error
        elif self.X_test is None or self.y_test is None:
             logger.info("ℹ️ Test dataset (X_test or y_test) is not provided. Skipping test predictions.")
        elif self.X_test.shape[0] == 0:
             logger.info("ℹ️ Test dataset (X_test) is empty. Skipping test predictions.")


        # --- Validation Set Predictions ---
        if self.X_val is not None and self.y_val is not None and self.X_val.shape[0] > 0:
            try:
                logger.info("⏳ Running predictions on the validation dataset...")
                self.y_pred_val_scaled = self.model.predict(self.X_val, verbose=0)
                self.y_pred_val_rescaled = self.scaler_y.inverse_transform(self.y_pred_val_scaled)
                self.y_val_rescaled = self.scaler_y.inverse_transform(self.y_val) # y_val should be scaled form
                logger.info("✅ Validation dataset prediction and rescaling completed.")
            except Exception as e:
                logger.error(f"❌ Error during validation prediction or rescaling: {e}", exc_info=True)
                self.y_pred_val_rescaled, self.y_val_rescaled = None, None # Reset on error
        elif self.X_val is None or self.y_val is None:
            logger.info("ℹ️ Validation dataset (X_val or y_val) is not provided. Skipping validation predictions.")
        elif self.X_val.shape[0] == 0:
            logger.info("ℹ️ Validation dataset (X_val) is empty. Skipping validation predictions.")


        if self.y_pred_test_rescaled is None and self.y_pred_val_rescaled is None:
            logger.warning("⚠️ No predictions were made for either test or validation sets.")
        else:
            logger.info("🏁 Prediction and rescaling process successfully completed for available datasets.")

        return self.y_pred_test_rescaled, self.y_test_rescaled, self.y_pred_val_rescaled, self.y_val_rescaled

    def calculate_metrics(self) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[Tuple[float, float, float, float]]]:
        """
        Calculates MAE, MSE, RMSE, and R² score for test and/or validation data.

        Assumes `predict()` has been called and rescaled predictions/actuals are available
        as instance attributes. Results are stored in instance attributes.

        Returns:
            Tuple containing:
                - test_metrics (Optional[Tuple[float, float, float, float]]):
                    (MAE, MSE, RMSE, R2) for the test set. None if test data was not evaluated.
                - val_metrics (Optional[Tuple[float, float, float, float]]):
                    (MAE, MSE, RMSE, R2) for the validation set. None if validation data was not evaluated.
        """
        logger.info("⚙️ Calculating evaluation metrics for available datasets...")
        test_metrics_results: Optional[Tuple[float, float, float, float]] = None
        val_metrics_results: Optional[Tuple[float, float, float, float]] = None

        # --- Test Metrics ---
        if self.y_test_rescaled is not None and self.y_pred_test_rescaled is not None:
            try:
                self.mae_test = mean_absolute_error(self.y_test_rescaled, self.y_pred_test_rescaled)
                self.mse_test = mean_squared_error(self.y_test_rescaled, self.y_pred_test_rescaled)
                self.rmse_test = np.sqrt(self.mse_test)
                self.r2_test = r2_score(self.y_test_rescaled, self.y_pred_test_rescaled)
                test_metrics_results = (self.mae_test, self.mse_test, self.rmse_test, self.r2_test)
                logger.info(f"🔬 Test Metrics: MAE={self.mae_test:.4f}, MSE={self.mse_test:.4f}, RMSE={self.rmse_test:.4f}, R²={self.r2_test:.4f}")
            except Exception as e:
                logger.error(f"❌ Error calculating test metrics: {e}", exc_info=True)
        else:
            logger.info("ℹ️ Test data/predictions not available for metric calculation.")

        # --- Validation Metrics ---
        if self.y_val_rescaled is not None and self.y_pred_val_rescaled is not None:
            try:
                self.mae_val = mean_absolute_error(self.y_val_rescaled, self.y_pred_val_rescaled)
                self.mse_val = mean_squared_error(self.y_val_rescaled, self.y_pred_val_rescaled)
                self.rmse_val = np.sqrt(self.mse_val)
                self.r2_val = r2_score(self.y_val_rescaled, self.y_pred_val_rescaled)
                val_metrics_results = (self.mae_val, self.mse_val, self.rmse_val, self.r2_val)
                logger.info(f"🔬 Validation Metrics: MAE={self.mae_val:.4f}, MSE={self.mse_val:.4f}, RMSE={self.rmse_val:.4f}, R²={self.r2_val:.4f}")
            except Exception as e:
                logger.error(f"❌ Error calculating validation metrics: {e}", exc_info=True)
        else:
            logger.info("ℹ️ Validation data/predictions not available for metric calculation.")

        return test_metrics_results, val_metrics_results


    def _save_plot(self, fig: plt.Figure, plot_name: str, sub_dir: Optional[str] = None):
        """Helper function to save a Matplotlib figure."""
        if fig is None:
            logger.warning(f"Attempted to save a None figure for {plot_name}. Skipping.")
            return

        save_dir = self.run_dir
        if sub_dir:
            save_dir = os.path.join(self.run_dir, sub_dir)
            os.makedirs(save_dir, exist_ok=True)

        plot_path = os.path.join(save_dir, plot_name)
        try:
            fig.savefig(plot_path)
            plt.close(fig) # Close the figure to free memory
            logger.info(f"📊 Plot saved successfully to: {plot_path}")
        except Exception as e:
            logger.error(f"❌ Error saving plot to {plot_path}: {e}", exc_info=True)


    def plot_loss(self, history: tf.keras.callbacks.History) -> Optional[plt.Figure]:
        """
        Plots training and validation loss from Keras history.

        Args:
            history (tf.keras.callbacks.History): Keras History object from model training.

        Returns:
            Optional[plt.Figure]: The Matplotlib figure object, or None if plotting failed.
                                  The figure is also saved to a file.
        """
        logger.debug("Plotting loss over epochs.")
        if not hasattr(history, 'history') or not history.history:
            logger.error("Keras history object is invalid or empty. Cannot plot loss.")
            return None
        if 'loss' not in history.history or 'val_loss' not in history.history:
            logger.error("'loss' or 'val_loss' keys not found in history. Cannot plot loss.")
            return None

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(history.history['loss'], label='Train Loss', color='blue')
        ax.plot(history.history['val_loss'], label='Validation Loss', color='orange')
        ax.set_title('Model Loss During Training')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper right')
        ax.grid(True)
        logger.debug("Loss plot created.")
        self._save_plot(fig, "loss_evolution_plot.png")
        return fig # Return fig in case it's needed by the caller, though it's closed after saving


    def plot_metric_evolution(self, history: tf.keras.callbacks.History, metric_name: str, plot_filename: str):
        """
        Plots a specific training and validation metric (e.g., MAE, MSE) over epochs from Keras history.

        Args:
            history (tf.keras.callbacks.History): Keras History object.
            metric_name (str): The base name of the metric (e.g., 'mae', 'mse').
                               The function will look for 'metric_name' and 'val_' + 'metric_name'.
            plot_filename (str): Filename for the saved plot (e.g., 'mae_plot.png').
        """
        logger.debug(f"Plotting {metric_name.upper()} over epochs.")
        train_metric_key = metric_name
        val_metric_key = f'val_{metric_name}'

        if not hasattr(history, 'history') or not history.history:
            logger.error(f"Keras history object is invalid or empty. Cannot plot {metric_name.upper()}.")
            return
        if train_metric_key not in history.history or val_metric_key not in history.history:
            logger.error(f"'{train_metric_key}' or '{val_metric_key}' not found in history. Cannot plot {metric_name.upper()}.")
            return

        train_values = history.history[train_metric_key]
        val_values = history.history[val_metric_key]

        # If metric is MSE, also plot RMSE
        if metric_name == 'mse':
            train_rmse = np.sqrt(train_values)
            val_rmse = np.sqrt(val_values)
            fig_rmse, ax_rmse = plt.subplots(figsize=(12, 6))
            ax_rmse.plot(train_rmse, label='Train RMSE', color='green')
            ax_rmse.plot(val_rmse, label='Validation RMSE', color='red')
            ax_rmse.set_title('Root Mean Squared Error (RMSE) During Training')
            ax_rmse.set_ylabel('RMSE')
            ax_rmse.set_xlabel('Epoch')
            ax_rmse.legend(loc='upper right')
            ax_rmse.grid(True)
            self._save_plot(fig_rmse, "rmse_evolution_plot.png")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(train_values, label=f'Train {metric_name.upper()}', color='blue')
        ax.plot(val_values, label=f'Validation {metric_name.upper()}', color='orange')
        ax.set_title(f'{metric_name.upper()} During Training')
        ax.set_ylabel(metric_name.upper())
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper right')
        ax.grid(True)
        self._save_plot(fig, plot_filename)
        logger.debug(f"{metric_name.upper()} evolution plot saved to {plot_filename}.")


    def plot_predictions(self) -> Tuple[Optional[plt.Figure], Optional[plt.Figure]]:
        """
        Plots actual vs. predicted values for test and/or validation sets.

        Assumes `predict()` has been called and rescaled predictions/actuals are available.
        Plots are saved to files.

        Returns:
            Tuple[Optional[plt.Figure], Optional[plt.Figure]]:
                - fig_test_pred: Figure for test set predictions (or None).
                - fig_val_pred: Figure for validation set predictions (or None).
                Figures are closed after saving.
        """
        logger.debug("Plotting actual vs predicted values.")
        fig_test_pred: Optional[plt.Figure] = None
        fig_val_pred: Optional[plt.Figure] = None

        # --- Test Set Prediction Plot ---
        if self.y_test_rescaled is not None and self.y_pred_test_rescaled is not None:
            fig_test_pred, ax_test = plt.subplots(figsize=(14, 7))
            ax_test.plot(self.y_test_rescaled, label='Actual Test Data', color='blue', alpha=0.8)
            ax_test.plot(self.y_pred_test_rescaled, label='Predicted Test Data', color='red', linestyle='--', alpha=0.8)
            ax_test.set_title('Actual vs. Predicted Test Data')
            ax_test.set_xlabel('Sample Index')
            ax_test.set_ylabel('Value (Original Scale)')
            ax_test.legend()
            ax_test.grid(True)
            self._save_plot(fig_test_pred, 'test_predictions_vs_actual.png')
        else:
            logger.info("ℹ️ Test data/predictions not available for plotting.")

        # --- Validation Set Prediction Plot ---
        if self.y_val_rescaled is not None and self.y_pred_val_rescaled is not None:
            fig_val_pred, ax_val = plt.subplots(figsize=(14, 7))
            ax_val.plot(self.y_val_rescaled, label='Actual Validation Data', color='green', alpha=0.8)
            ax_val.plot(self.y_pred_val_rescaled, label='Predicted Validation Data', color='orange', linestyle='--', alpha=0.8)
            ax_val.set_title('Actual vs. Predicted Validation Data')
            ax_val.set_xlabel('Sample Index')
            ax_val.set_ylabel('Value (Original Scale)')
            ax_val.legend()
            ax_val.grid(True)
            self._save_plot(fig_val_pred, 'validation_predictions_vs_actual.png')
        else:
            logger.info("ℹ️ Validation data/predictions not available for plotting.")

        return fig_test_pred, fig_val_pred


    def plot_error_distribution(self) -> Tuple[Optional[plt.Figure], Optional[plt.Figure]]:
        """
        Plots the distribution of prediction errors for test and/or validation sets.

        Assumes `predict()` has been called. Errors are calculated as (actual - predicted).
        Plots are saved to files.

        Returns:
            Tuple[Optional[plt.Figure], Optional[plt.Figure]]:
                - fig_test_error: Figure for test set error distribution (or None).
                - fig_val_error: Figure for validation set error distribution (or None).
                Figures are closed after saving.
        """
        logger.debug("Plotting distribution of prediction errors.")
        fig_test_error: Optional[plt.Figure] = None
        fig_val_error: Optional[plt.Figure] = None

        # --- Test Set Error Distribution ---
        if self.y_test_rescaled is not None and self.y_pred_test_rescaled is not None:
            errors_test = self.y_test_rescaled.flatten() - self.y_pred_test_rescaled.flatten()
            fig_test_error, ax_test = plt.subplots(figsize=(10, 6))
            ax_test.hist(errors_test, bins=50, color='purple', edgecolor='black', alpha=0.7)
            ax_test.set_title('Distribution of Test Prediction Errors (Actual - Predicted)')
            ax_test.set_xlabel('Prediction Error')
            ax_test.set_ylabel('Frequency')
            ax_test.grid(True)
            self._save_plot(fig_test_error, 'test_error_distribution.png')
        else:
            logger.info("ℹ️ Test data/predictions not available for error distribution plot.")

        # --- Validation Set Error Distribution ---
        if self.y_val_rescaled is not None and self.y_pred_val_rescaled is not None:
            errors_val = self.y_val_rescaled.flatten() - self.y_pred_val_rescaled.flatten()
            fig_val_error, ax_val = plt.subplots(figsize=(10, 6))
            ax_val.hist(errors_val, bins=50, color='teal', edgecolor='black', alpha=0.7)
            ax_val.set_title('Distribution of Validation Prediction Errors (Actual - Predicted)')
            ax_val.set_xlabel('Prediction Error')
            ax_val.set_ylabel('Frequency')
            ax_val.grid(True)
            self._save_plot(fig_val_error, 'validation_error_distribution.png')
        else:
            logger.info("ℹ️ Validation data/predictions not available for error distribution plot.")

        return fig_test_error, fig_val_error


    def plot_r2_bar(self, r2_value: Optional[float], dataset_name: str = "Test"):
        """
        Plots the R² score as a bar chart.

        Args:
            r2_value (Optional[float]): The R² score to plot.
            dataset_name (str): Name of the dataset (e.g., "Test", "Validation") for title and filename.
        """
        logger.debug(f"Plotting R² score for {dataset_name} set.")
        if r2_value is None or np.isnan(r2_value):
            logger.warning(f"R² value for {dataset_name} is None or NaN. Skipping R² plot.")
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar([f'{dataset_name} R² Score'], [r2_value], color='#4682B4', edgecolor='black', width=0.5)
        ax.axhline(y=0.0, color='black', linestyle='-', linewidth=0.8) # Line at R2=0
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, label='Baseline (0.5)')
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=0.8, label='Perfect Score (1.0)')

        # Adjust y-limits based on R2 value to make plot informative
        min_y_lim = min(-0.1, r2_value - 0.2) if r2_value < 0 else 0
        max_y_lim = max(1.1, r2_value + 0.2)
        ax.set_ylim(min_y_lim, max_y_lim)

        ax.set_title(f'R² Score of the Model on {dataset_name} Data', fontsize=14, fontweight='bold')
        ax.set_ylabel('R² Value', fontsize=12)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend(loc='best', fontsize=10) # Changed loc to 'best'
        # Display R2 value on the bar
        ax.text(0, r2_value + (0.03 * np.sign(r2_value) if r2_value !=0 else 0.03), f"{r2_value:.5f}",
                ha='center', va='bottom' if r2_value >= 0 else 'top',
                fontsize=12, fontweight='bold', color='black')
        ax.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

        self._save_plot(fig, f'{dataset_name.lower()}_r2_score_plot.png')
        logger.info(f"R² plot for {dataset_name} set saved.")


    def save_metrics_to_file(self, file_name: str = "evaluation_metrics.txt"):
        """
        Saves all calculated metrics (MAE, MSE, RMSE, R²) for test and validation sets
        to a text file in the `run_dir`.

        Args:
            file_name (str): Name of the text file to save metrics.
        """
        metrics_path = os.path.join(self.run_dir, file_name)
        try:
            with open(metrics_path, 'w') as f:
                f.write("Evaluation Metrics\n")
                f.write("=" * 20 + "\n")
                if self.mae_test is not None: # Check if test metrics were calculated
                    f.write("Test Set:\n")
                    f.write(f"  Mean Absolute Error (MAE): {self.mae_test:.6f}\n")
                    f.write(f"  Mean Squared Error (MSE): {self.mse_test:.6f}\n")
                    f.write(f"  Root Mean Squared Error (RMSE): {self.rmse_test:.6f}\n")
                    f.write(f"  R² Score: {self.r2_test:.6f}\n")
                else:
                    f.write("Test Set: Metrics not calculated or not available.\n")

                f.write("\nValidation Set:\n")
                if self.mae_val is not None: # Check if validation metrics were calculated
                    f.write(f"  Mean Absolute Error (MAE): {self.mae_val:.6f}\n")
                    f.write(f"  Mean Squared Error (MSE): {self.mse_val:.6f}\n")
                    f.write(f"  Root Mean Squared Error (RMSE): {self.rmse_val:.6f}\n")
                    f.write(f"  R² Score: {self.r2_val:.6f}\n")
                else:
                    f.write("Validation Set: Metrics not calculated or not available.\n")
            logger.info(f"📋 Evaluation metrics saved to: {metrics_path}")
        except IOError as e:
            logger.error(f"❌ Error saving metrics to {metrics_path}: {e}", exc_info=True)

    # Original print_metrics method - can be kept if direct printing is needed,
    # but save_metrics_to_file is generally more useful for record-keeping.
    # def print_metrics(self, mae: float, mse: float, rmse: float, r2: float) -> None:
    #     """Prints a set of metrics to the logger."""
    #     logger.info(f"Mean Absolute Error (MAE): {mae}")
    #     logger.info(f"Mean Squared Error (MSE): {mse}")
    #     logger.info(f"Root Mean Squared Error (RMSE): {rmse}")
    #     logger.info(f"R² Score: {r2}")