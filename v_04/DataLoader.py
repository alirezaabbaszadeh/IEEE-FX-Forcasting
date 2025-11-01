import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import logging
from typing import Tuple, List, Optional

# It's good practice to get the logger by the module name
logger = logging.getLogger(__name__)
# Configure logging level as needed, or add a NullHandler for library use.
# For this project, INFO or DEBUG might be useful during development.
# logger.setLevel(logging.INFO)


class DataLoader:
    """
    Handles loading, preprocessing, and splitting time series data for model training and evaluation.

    The class takes a CSV file path, desired time steps for sequence creation,
    and ratios for splitting data into training, validation, and test sets.
    It scales the features and target variable, creates time-windowed sequences,
    and returns the processed data ready for model input.
    """

    def __init__(self,
                 file_path: str,
                 time_steps: int = 3,
                 train_ratio: float = 0.94,
                 val_ratio: float = 0.03,
                 test_ratio: float = 0.03):
        """
        Initializes the DataLoader.

        Args:
            file_path (str): Path to the CSV file containing the time series data.
                             The CSV should have columns like 'Open', 'High', 'Low', 'Close'.
            time_steps (int): The number of past time steps to use as input features
                              for predicting the next time step.
            train_ratio (float): Proportion of data to use for the training set.
            val_ratio (float): Proportion of data to use for the validation set.
            test_ratio (float): Proportion of data to use for the test set.

        Raises:
            ValueError: If the sum of train_ratio, val_ratio, and test_ratio is not close to 1.0.
        """
        self.file_path: str = file_path
        self.time_steps: int = time_steps
        self.train_ratio: float = train_ratio
        self.val_ratio: float = val_ratio
        self.test_ratio: float = test_ratio

        if not np.isclose(self.train_ratio + self.val_ratio + self.test_ratio, 1.0):
            raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must be equal to 1.0.")

        # Initialize scalers for features (X) and target (y)
        self.scaler_X: StandardScaler = StandardScaler()
        self.scaler_y: StandardScaler = StandardScaler()

        logger.debug(
            "DataLoader initialized: file_path=%s, time_steps=%d, train_ratio=%.2f, val_ratio=%.2f, test_ratio=%.2f",
            self.file_path, self.time_steps, self.train_ratio, self.val_ratio, self.test_ratio
        )

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from the specified CSV file.

        Performs basic cleaning by dropping rows with NaN values.

        Returns:
            pd.DataFrame: DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If the CSV file specified in file_path does not exist.
        """
        if not os.path.exists(self.file_path):
            logger.error("File not found at path: %s", self.file_path)
            raise FileNotFoundError(f"File not found at path: {self.file_path}")

        logger.info("Loading data from %s", self.file_path)
        data: pd.DataFrame = pd.read_csv(self.file_path)

        # Handle missing values
        if data.isnull().values.any():
            logger.warning("Data contains NaN values. Dropping NaN rows.")
            data = data.dropna()

        logger.info("Data loaded successfully. Data shape: %s", data.shape)
        return data

    def _extract_features_target(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts features (X) and target (y) from the DataFrame.

        Features are 'Open', 'High', 'Low', 'Close'.
        Target is 'Close'.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the feature array (X)
                                           and the target array (y). y is reshaped to be 2D.
        """
        # Define feature columns and target column
        # 'VolumeASK', 'VolumeBID' were commented out in the original code,
        # so they are not included as features here.
        feature_columns: List[str] = ['Open', 'High', 'Low', 'Close']
        target_column: str = 'Close'

        X: np.ndarray = df[feature_columns].values
        y: np.ndarray = df[target_column].values.reshape(-1, 1)  # y needs to be 2D for the scaler

        logger.debug("Features extracted using columns: %s. Target extracted from column: '%s'.",
                     feature_columns, target_column)
        return X, y

    def create_sequences(self, X_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms time series data into sequences of a specified number of time steps.

        Each sequence in Xs corresponds to a target value in ys.
        Xs shape: (num_samples, time_steps, num_features)
        ys shape: (num_samples, 1)

        Args:
            X_data (np.ndarray): The input feature data (scaled).
            y_data (np.ndarray): The input target data (scaled and 2D).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the sequenced feature array (Xs)
                                           and the corresponding sequenced target array (ys).
        """
        logger.info("Creating sequences with time_steps=%d", self.time_steps)
        Xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []

        # Ensure data length is sufficient for at least one sequence
        if len(X_data) <= self.time_steps:
            logger.warning(
                f"Input data length ({len(X_data)}) is not sufficient to create "
                f"sequences with time_steps={self.time_steps}. Returning empty arrays."
            )
            num_features = X_data.shape[1] if X_data.ndim > 1 and X_data.shape[0] > 0 else 1
            # Return empty arrays with the correct number of dimensions
            Xs_empty = np.empty((0, self.time_steps, num_features))
            ys_empty = np.empty((0, 1)) # y_data is expected to be 2D, so ys_empty should also be
            return Xs_empty, ys_empty

        # Iterate through the data to create sequences
        for i in range(len(X_data) - self.time_steps):
            # The sequence X_data[i:(i + self.time_steps)] is used to predict y_data[i + self.time_steps]
            Xs.append(X_data[i:(i + self.time_steps)])
            ys.append(y_data[i + self.time_steps])

        Xs_np: np.ndarray = np.array(Xs)
        ys_np: np.ndarray = np.array(ys)

        logger.info("Sequences created successfully.")
        logger.debug("Shape of sequenced features (Xs): %s, Shape of sequenced target (ys): %s",
                     Xs_np.shape, ys_np.shape)
        return Xs_np, ys_np

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
        """
        Main method to load, preprocess, split, and sequence the data.

        Steps:
        1. Loads data from CSV.
        2. Splits data chronologically into training, validation, and test sets.
        3. Extracts features and target for each set.
        4. Fits scalers (StandardScaler) on the training set features and target.
        5. Scales all sets (training, validation, test) using the fitted scalers.
        6. Creates time-windowed sequences for each set.

        Returns:
            Tuple containing:
                - X_train (np.ndarray): Sequenced training features.
                - X_val (np.ndarray): Sequenced validation features.
                - X_test (np.ndarray): Sequenced test features.
                - y_train (np.ndarray): Sequenced training target.
                - y_val (np.ndarray): Sequenced validation target.
                - y_test (np.ndarray): Sequenced test target.
                - scaler_y (StandardScaler): The fitted scaler for the target variable,
                                             useful for inverse transforming predictions.
        Raises:
            ValueError: If the loaded data is empty or if the training set is empty after splitting.
        """
        logger.info("Starting the full data loading and preprocessing pipeline.")
        full_data_df: pd.DataFrame = self.load_data()

        if full_data_df.empty:
            logger.error("Loaded data is empty. Cannot proceed.")
            raise ValueError("Loaded data is empty.")

        # Chronological split of the raw DataFrame
        n: int = len(full_data_df)
        train_end_idx: int = int(self.train_ratio * n)
        val_end_idx: int = int((self.train_ratio + self.val_ratio) * n)

        df_train: pd.DataFrame = full_data_df.iloc[:train_end_idx]
        df_val: pd.DataFrame = full_data_df.iloc[train_end_idx:val_end_idx]
        df_test: pd.DataFrame = full_data_df.iloc[val_end_idx:]

        logger.info(f"Data split into sets: Training ({len(df_train)} rows), "
                    f"Validation ({len(df_val)} rows), Test ({len(df_test)} rows).")

        # Extract features and target for each raw dataset
        X_train_raw, y_train_raw = self._extract_features_target(df_train)
        X_val_raw, y_val_raw = self._extract_features_target(df_val)
        X_test_raw, y_test_raw = self._extract_features_target(df_test)

        # Ensure training data is not empty before proceeding with scaling
        if X_train_raw.shape[0] == 0:
            logger.error("Training dataset is empty after splitting. Check data split ratios or initial data size.")
            raise ValueError("Training dataset is empty after splitting. Check data split ratios or initial data size.")

        # Fit scalers exclusively on training data to prevent data leakage
        logger.info("Fitting scalers on the training data.")
        self.scaler_X.fit(X_train_raw)
        self.scaler_y.fit(y_train_raw)  # y_train_raw is already (N_train, 1)

        # Scale features and target for all sets using the fitted scalers
        logger.info("Scaling training, validation, and test sets.")
        X_train_scaled: np.ndarray = self.scaler_X.transform(X_train_raw)
        y_train_scaled: np.ndarray = self.scaler_y.transform(y_train_raw) # Output shape: (N_train, 1)

        # Handle potentially empty validation or test sets before scaling
        X_val_scaled: np.ndarray = self.scaler_X.transform(X_val_raw) if X_val_raw.shape[0] > 0 else np.array([])
        y_val_scaled: np.ndarray = self.scaler_y.transform(y_val_raw) if y_val_raw.shape[0] > 0 else np.array([])

        X_test_scaled: np.ndarray = self.scaler_X.transform(X_test_raw) if X_test_raw.shape[0] > 0 else np.array([])
        y_test_scaled: np.ndarray = self.scaler_y.transform(y_test_raw) if y_test_raw.shape[0] > 0 else np.array([])

        # Ensure y_scaled arrays maintain 2D shape (n_samples, 1) if not empty, as create_sequences expects this.
        if y_val_scaled.ndim == 1 and y_val_scaled.shape[0] > 0: y_val_scaled = y_val_scaled.reshape(-1, 1)
        if y_test_scaled.ndim == 1 and y_test_scaled.shape[0] > 0: y_test_scaled = y_test_scaled.reshape(-1, 1)


        # Create sequences from the scaled data
        logger.info("Creating sequences for training, validation, and test sets.")
        X_train, y_train = self.create_sequences(X_train_scaled, y_train_scaled)

        # Determine num_features_X based on available data, preferring training data
        if X_train_scaled.ndim > 1 and X_train_scaled.shape[0] > 0:
            num_features_X = X_train_scaled.shape[1]
        elif X_val_scaled.ndim > 1 and X_val_scaled.shape[0] > 0:
            num_features_X = X_val_scaled.shape[1]
        elif X_test_scaled.ndim > 1 and X_test_scaled.shape[0] > 0:
            num_features_X = X_test_scaled.shape[1]
        else: # Fallback if all are empty (though earlier checks should prevent this for train)
            logger.warning("Could not determine num_features_X as all scaled feature sets are empty or 1D.")
            num_features_X = 1 # Default to 1 if undetermined, though sequences might be invalid

        # Create sequences for validation set, handling empty or small sets
        if X_val_scaled.shape[0] > self.time_steps : # Ensure there's enough data for at least one sequence
            X_val, y_val = self.create_sequences(X_val_scaled, y_val_scaled)
        else:
            logger.warning("Validation set is empty or too small to create sequences. Creating empty arrays.")
            X_val = np.empty((0, self.time_steps, num_features_X))
            y_val = np.empty((0, 1)) # Ensure y_val is 2D

        # Create sequences for test set, handling empty or small sets
        if X_test_scaled.shape[0] > self.time_steps: # Ensure there's enough data for at least one sequence
            X_test, y_test = self.create_sequences(X_test_scaled, y_test_scaled)
        else:
            logger.warning("Test set is empty or too small to create sequences. Creating empty arrays.")
            X_test = np.empty((0, self.time_steps, num_features_X))
            y_test = np.empty((0, 1)) # Ensure y_test is 2D


        logger.info("Data loading and preprocessing pipeline completed successfully.")
        # y_train, y_val, y_test should have shape (num_sequences, 1)
        return X_train, X_val, X_test, y_train, y_val, y_test, self.scaler_y