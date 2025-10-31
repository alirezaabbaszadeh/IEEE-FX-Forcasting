# Legacy DataLoader Catalogue

This catalogue summarises the shared behaviour observed across the ten legacy
experiment packages (`v_01` … `v_10`) prior to consolidating them behind the
shared loader.

## Parameter defaults

All legacy loaders expose the same constructor signature:

| Parameter    | Default | Notes |
|--------------|---------|-------|
| `time_steps` | `3`     | Sliding-window length used when creating sequences. |
| `train_ratio`| `0.94`  | Fraction of chronological rows assigned to the training split. |
| `val_ratio`  | `0.03`  | Fraction assigned to validation. |
| `test_ratio` | `0.03`  | Fraction assigned to the hold-out evaluation set. |

Each version validates that the ratios sum to 1.0 before continuing.

## Feature and target columns

Every loader extracts the same columns from the raw CSV files:

- Features: `Open`, `High`, `Low`, `Close`
- Target: `Close`

Feature and target arrays are scaled independently with dedicated
`StandardScaler` instances fit on the training split only.

## Split and sequencing logic

Rows are sorted chronologically (as provided in the files) and split via
half-open index bounds `[start, end)` computed from the configured ratios. The
sliding-window generator yields:

- `X_*` tensors with shape `(num_sequences, time_steps, num_features)`
- `y_*` tensors with shape `(num_sequences, 1)`

When a partition contains fewer rows than `time_steps + 1`, an empty array with
the correct dimensionality is returned for both features and target.

## Exceptions and uncovered behaviours

During the audit no version diverged from the defaults above—there are no
additional feature sets, DST adjustments, or split deviations present in the
checked-in loaders. The consolidated implementation therefore focuses on the
shared path while exposing configuration flags (feature selection, timestamp
sorting, duplicate removal) so future provenance gaps can be addressed without
forking the loader again.
