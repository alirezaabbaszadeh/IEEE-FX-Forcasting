# P2.M1 Configuration Schema

This document defines the canonical configuration schema consumed by
`bin/run_experiment.py` when delegating experiments to the shared
`TimeSeriesOrchestrator`.  The schema is expressed in YAML (JSON syntax is
allowed because JSON is a valid subset of YAML) and is stored alongside each
legacy experiment under `v_*/default_config.yaml`.

## Top-level structure

```yaml
version: v_08                       # Optional descriptor captured as metadata
file_path: ./csv/....csv            # Dataset path (resolved relative to the config)
base_dir: ./IEEE_TNNLS_Runs_V8      # Run directory root

# Sections forwarded directly into the orchestrator config
data:
  time_steps: 3                     # Passed to the DataLoader factory
  train_ratio: 0.96
  val_ratio: 0.02
  test_ratio: 0.02

model:
  block_configs: []                 # Injected into the ModelBuilder constructor
  optimizer_lr: 0.01                # Any hyperparameter accepted by the builder

training:
  epochs: 60                        # Trainer arguments
  batch_size: 5000

callbacks:
  early_stopping_patience: 20       # Trainer callback knobs
  reduce_lr_patience: 1
  reduce_lr_factor: 0.1
  min_lr: 5.0e-7

metadata:
  legacy_version: v_08              # Persisted in the hyperparameter snapshot

dependencies:                       # Factories used to instantiate dependencies
  data_loader: v_08.DataLoader.DataLoader
  model_builder: v_08.ModelBuilder.ModelBuilder
  trainer: v_08.Trainer.Trainer
  evaluator: v_08.Evaluator.Evaluator
  history_manager: v_08.HistoryManager.HistoryManager

runtime:                            # Runner specific behaviour
  seed: 42
  mixed_precision:
    enabled: true
    policy: mixed_float16

cli:                                # Optional CLI flag translations
  flags:
    epochs: { path: training.epochs }
    batch_size: { path: training.batch_size }
    optimizer_lr: { path: model.optimizer_lr }
  toggles:
    disable_mixed_precision: { path: runtime.mixed_precision.enabled, value: false }
```

`file_path` and `base_dir` are resolved relative to the configuration file before
being passed into the orchestrator.  All nested dictionaries (`data`, `model`,
`training`, `callbacks`, `metadata`) are forwarded unchanged, enabling full
coverage of version-specific hyperparameters.

## CLI behaviour

The new runner supports two mechanisms for overrides:

1. Global overrides available for every version
   - `--config`: select a configuration file (required)
   - `--set section.key=value`: arbitrarily mutate nested values using dot
     notation (e.g. `--set training.epochs=100`)
   - `--seed` / `--mixed-precision` / `--no-mixed-precision` /
     `--precision-policy`: override runtime execution settings
   - `--file-path` / `--base-dir`: shortcuts for the two most common paths
2. Shared aliases defined in `src/core/versioning/cli_schema.json` with optional
   per-config overrides under `cli.flags`/`cli.toggles`
   - Flags map the historical CLI (e.g. `--epochs`) to a concrete config path
   - Toggles map boolean flags (e.g. `--disable_mixed_precision`) to a constant
     override (`true`/`false`)

The shared schema provides canonical paths for every legacy flag while the
normalisation pass mirrors values onto the legacy parameter names expected by
the version-specific factories.【F:src/core/versioning/cli_schema.json†L1-L95】【F:src/core/versioning/configuration.py†L1-L188】

After resolution all overrides are merged into the configuration and written to
`hyperparameters_and_summary.json` by the orchestrator for reproducibility.

## Mapping into the orchestrator

`bin/run_experiment.py` performs the following transformations before invoking
`TimeSeriesOrchestrator`:

1. Load and normalise configuration values (resolving relative paths and
   merging overrides)
2. Instantiate `OrchestratorConfig` with
   - `file_path`, `base_dir`
   - `data`, `model`, `training`, `callbacks`, `metadata`
3. Instantiate `OrchestratorDependencies` using the import paths declared in the
   `dependencies` section
4. Configure runtime seed/mixed precision according to the `runtime` block
5. Execute `TimeSeriesOrchestrator.run()` which wires the legacy factories into
   the shared training/evaluation lifecycle

The schema ensures that every original experiment parameter (time steps,
train/validation/test splits, optimiser settings, output directories, etc.) is
captured declaratively, making historical runs reproducible via a single entry
point.
