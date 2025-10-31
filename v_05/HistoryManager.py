# HistoryManager.py
import json
import os
import matplotlib.pyplot as plt # For plot_history method
import logging
import tensorflow as tf # For type hinting Keras History object
from typing import Optional, Dict, List # For type hinting

# Get logger for this module
logger = logging.getLogger(__name__)

class HistoryManager:
    """
    Manages loading and saving of Keras model training history.
    The history is typically stored in a JSON file.
    """

    def __init__(self, history_path: str):
        """
        Initializes the HistoryManager with the path to the history JSON file.

        Args:
            history_path (str): The file path where the JSON training history is/will be stored.
        """
        self.history_path: str = history_path
        self.history: Optional[Dict[str, List[float]]] = None # Stores the loaded history dictionary

    def save_history(self, keras_history_obj: tf.keras.callbacks.History):
        """
        Saves the training history from a Keras History object to a JSON file.
        Ensures all metric values are converted to standard Python floats for serialization.

        Args:
            keras_history_obj (tf.keras.callbacks.History): The Keras History object
                                                            (returned by model.fit()).
        """
        if not hasattr(keras_history_obj, 'history') or not isinstance(keras_history_obj.history, dict):
            logger.error("Invalid Keras History object provided or its 'history' attribute is not a dictionary. Cannot save.")
            return

        history_dict_to_save: Dict[str, List[float]] = {}
        for key, values_list in keras_history_obj.history.items():
            try:
                # Attempt to convert each value in the list to float
                history_dict_to_save[key] = [float(v) for v in values_list]
            except (TypeError, ValueError) as e:
                logger.warning(
                    f"Could not convert all values in history key '{key}' to float "
                    f"for JSON serialization: {e}. Using raw values for this key, which might cause issues."
                )
                history_dict_to_save[key] = values_list # Fallback, but might fail during json.dump

        try:
            # Create directory if it doesn't exist for self.history_path
            os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
            with open(self.history_path, 'w') as f:
                json.dump(history_dict_to_save, f, indent=4)
            logger.info(f"💾 Training history saved successfully to: {self.history_path}")
        except IOError as e:
            logger.error(f"❌ Could not write history to {self.history_path}: {e}", exc_info=True)
        except TypeError as e:
            # This might happen if fallback values are not serializable
            logger.error(f"❌ TypeError during JSON dump for history: {e}. Data intended for save: {history_dict_to_save}", exc_info=True)

    def load_history(self) -> Optional[Dict[str, List[float]]]:
        """
        Loads the training history from the JSON file specified during initialization.

        Returns:
            Optional[Dict[str, List[float]]]: The loaded training history dictionary,
                                             or None if loading fails or file not found.
        """
        if not os.path.exists(self.history_path):
            logger.error(f"History file not found at: {self.history_path}")
            # raise FileNotFoundError(f"The history file at {self.history_path} was not found.") # Or return None
            return None

        try:
            with open(self.history_path, 'r') as f:
                self.history = json.load(f)
            logger.info(f"📚 Training history loaded successfully from: {self.history_path}")
            return self.history
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error decoding JSON from {self.history_path}: {e}", exc_info=True)
            self.history = None
            return None
        except IOError as e:
            logger.error(f"❌ Could not read history file at {self.history_path}: {e}", exc_info=True)
            self.history = None
            return None

    def plot_history(self, metrics_to_plot: Optional[List[str]] = None, save_plot: bool = False, show_plot: bool = True):
        """
        Plots specified metrics (e.g., 'loss', 'mae', 'val_loss', 'val_mae')
        from the loaded training history.

        Args:
            metrics_to_plot (Optional[List[str]]): A list of metric keys to plot (e.g., ['loss', 'val_loss', 'mae', 'val_mae']).
                                                  If None, attempts to plot 'loss' and 'val_loss'.
            save_plot (bool): If True, saves the plot to a file in the same directory as history_path.
            show_plot (bool): If True, displays the plot using plt.show().
        """
        if not self.history:
            logger.info("History not loaded. Attempting to load history before plotting.")
            if not self.load_history(): # If loading fails
                logger.error("Cannot plot history: Not loaded and loading failed.")
                return

        if metrics_to_plot is None:
            metrics_to_plot = ['loss', 'val_loss'] # Default metrics

        # Determine number of subplots needed (e.g., one for loss, one for MAE)
        # This example plots each metric pair (train/val) on a separate figure for clarity.
        # Or you could group them onto subplots of a single figure.

        for i in range(0, len(metrics_to_plot), 2): # Assuming metrics come in train/val pairs
            train_metric_key = metrics_to_plot[i]
            val_metric_key = metrics_to_plot[i+1] if (i+1) < len(metrics_to_plot) else None

            if train_metric_key not in self.history:
                logger.warning(f"Metric key '{train_metric_key}' not found in history. Skipping this plot.")
                continue
            if val_metric_key and val_metric_key not in self.history:
                logger.warning(f"Metric key '{val_metric_key}' not found in history. Plotting only '{train_metric_key}'.")
                val_metric_key = None # Plot only train metric

            base_metric_name = train_metric_key.replace("val_", "")

            plt.figure(figsize=(12, 6))
            plt.plot(self.history[train_metric_key], label=f'Train {base_metric_name.capitalize()}', color='blue')
            if val_metric_key:
                plt.plot(self.history[val_metric_key], label=f'Validation {base_metric_name.capitalize()}', color='orange')

            plt.title(f'Model {base_metric_name.capitalize()} During Training')
            plt.ylabel(base_metric_name.capitalize())
            plt.xlabel('Epoch')
            plt.legend(loc='upper right')
            plt.grid(True)

            plot_title = f"history_{base_metric_name}_plot.png"
            if save_plot:
                plot_save_path = os.path.join(os.path.dirname(self.history_path), plot_title)
                try:
                    plt.savefig(plot_save_path)
                    logger.info(f"📊 History plot for {base_metric_name} saved to: {plot_save_path}")
                except Exception as e:
                    logger.error(f"❌ Error saving history plot for {base_metric_name} to {plot_save_path}: {e}", exc_info=True)

            if show_plot:
                logger.info(f"Displaying history plot for {base_metric_name}. Close plot window to continue if in blocking mode.")
                plt.show()
            else:
                plt.close() # Close the figure if not shown, to free memory

        if not metrics_to_plot:
            logger.info("No metrics specified or found for plotting in history.")