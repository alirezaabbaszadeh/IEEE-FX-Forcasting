# P2.M2 Data Loader Specification

This specification captures the common behaviour implemented across the
historical `v_*/DataLoader.py` modules and defines the contract for the shared
loader that replaces them.

## Required inputs

* **file_path** (`str`): Absolute or relative path to a CSV file containing the
  raw FX time series. The loader validates that the path exists before reading.
* **time_steps** (`int`, default `3`): Number of look-back timesteps used to
  build input sequences. Sequences are created with a sliding window of length
  `time_steps` and predict the immediate next observation.
* **train_ratio / val_ratio / test_ratio** (`float`): Chronological split ratios
  that must sum to `1.0` (within floating-point tolerance). The loader floors
  the boundaries via `int(ratio * len(data))` to avoid shuffling rows.

Passing a configuration where the training slice ends up empty results in a
`ValueError`. Validation and test slices may be empty.

## Feature and target handling

* **Feature columns**: `['Open', 'High', 'Low', 'Close']`
* **Target column**: `'Close'`
* Feature extraction preserves ordering; the arrays keep shape `(N, 4)`
  directly from the CSV contents. The target is reshaped to `(N, 1)` before any
  scaler interaction so downstream models receive a consistent two-dimensional
  array.
* Rows containing `NaN` values are dropped prior to splitting, matching the
  defensive cleaning in every legacy variant.

## Scaling strategy

* Two independent `sklearn.preprocessing.StandardScaler` instances are used:
  one for the feature matrix (`scaler_X`) and one for the target (`scaler_y`).
* Both scalers are **fitted only on the training slice** to avoid information
  leakage. Validation and test partitions are transformed with the frozen
  training statistics.
* When a validation or test slice is empty the loader returns empty arrays while
  skipping the transform call entirely.
* The target scaler is exported for inference-time inverse transforms; the new
  shared loader also exposes the feature scaler to keep experimentation
  symmetrical.

## Chronological split semantics

* Rows are partitioned contiguously in the order they appear in the CSV.
  `train` receives the first `⌊train_ratio · N⌋` rows, `val` the next block, and
  `test` the remainder.
* Split boundaries are captured as `[start, end)` index pairs to guarantee the
  reproducibility of downstream artefacts (e.g. evaluation ranges, diagnostics).
* Sequence generation is performed **after** scaling to operate on normalised
  values. For each slice the loader iterates `i` from `0` to
  `len(slice) - time_steps - 1`, appending
  `slice[i : i + time_steps] -> slice[i + time_steps]` pairs. When a slice does
  not contain `time_steps + 1` rows the loader returns empty tensors with shapes
  `(0, time_steps, num_features)` for the features and `(0, 1)` for the target,
  mirroring the legacy guard rails.

## Returned artefacts

The shared implementation returns structured objects containing:

1. Sequenced datasets for the train/val/test partitions.
2. The fitted feature and target scalers (for serialisation and inference).
3. Split metadata describing the `[start, end)` indices used during the
   chronological partition.
4. The feature/target column names used to build the arrays.

These artefacts provide the reproducibility backbone required to swap in new
models without rewriting the preprocessing pipeline.
