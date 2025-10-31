# Shared Orchestrator Migration Notes

This document explains how to migrate each legacy experiment variant (`v_01` …
`v_10`) onto the shared `TimeSeriesOrchestrator` located at
`src/core/orchestrator.py`.  The orchestrator consolidates responsibilities that
were duplicated across the historical `TimeSeriesModel` implementations and
provides dependency-injection seams for version-specific behaviour.

## Responsibilities now provided by the orchestrator

The orchestrator centralises the following duties:

- Run directory and per-run logging bootstrap, including file handler lifecycle
  management.【F:src/core/orchestrator.py†L108-L149】【F:src/core/orchestrator.py†L212-L241】
- Canonical data → model → training → evaluation pipeline execution, including
  training summary aggregation and history persistence.【F:src/core/orchestrator.py†L151-L276】【F:src/core/orchestrator.py†L318-L367】
- Callback helpers used by trainers (epoch timing, per-epoch checkpoints, final
  exports).【F:src/core/orchestrator.py†L169-L206】
- Hyperparameter snapshotting that combines configuration, metadata, and
  computed training summary.【F:src/core/orchestrator.py†L209-L235】

Dependencies are provided via factories so that each variant can supply its own
`DataLoader`, `ModelBuilder`, `Trainer`, `Evaluator`, and `HistoryManager`
implementation (or register them with
`DependencyFactoryRegistry`).【F:src/core/orchestrator.py†L33-L102】

### Configuration and hooks

Each version should build an `OrchestratorConfig` populated with:

- `data`: Arguments forwarded to the data loader (e.g., `time_steps`,
  `train_ratio`, `val_ratio`, `test_ratio`).
- `model`: Builder-specific arguments (e.g., `block_configs`, custom hyperparameters).
- `training`: Trainer runtime knobs (`epochs`, `batch_size`).
- `callbacks`: Early stopping / LR scheduler settings.
- `metadata`: Any extra experiment descriptors to persist alongside the run.

Optional lifecycle hooks (`OrchestratorHooks`) allow bespoke behaviour before or
after major pipeline stages if a variant requires additional instrumentation.

## Version-by-version migration notes

The table below lists what moves into the orchestrator versus what remains
custom.  Unless stated otherwise, the shared orchestrator now owns run setup,
training/evaluation sequencing, history management, and model export.

### v_01

- **Move into orchestrator**: All pipeline steps previously defined in
  `TimeSeriesModel` (run directory creation, hyperparameter logging, model
  saving, evaluation sequencing).【F:v_01/MainClass.py†L24-L352】
- **Keep bespoke**:
  - Concrete factories pointing to `v_01/DataLoader`, `ModelBuilder`,
    `Trainer`, `Evaluator`, and `HistoryManager`.
  - Default configuration mirroring the original constructor values:
    `data={"time_steps": 60, "train_ratio": 0.94, "val_ratio": 0.03,
    "test_ratio": 0.03}`, `training={"epochs": 20, "batch_size": 1120}`, and
    `callbacks={"early_stopping_patience": 60, "reduce_lr_patience": 1,
    "reduce_lr_factor": 0.1, "min_lr": 5e-7}`.【F:v_01/MainClass.py†L50-L121】
  - Any additional metadata or plotting hooks can be injected via
    `metadata`/`OrchestratorHooks` if required.

### v_02

- **New in orchestrator**: The shared module now forwards `block_configs` to the
  model builder via `config.model["block_configs"]` instead of duplicating the
  call-site logic.【F:src/core/orchestrator.py†L252-L260】
- **Keep bespoke**: Same factories as v_01, with `config.model` populated so that
  the builder receives `block_configs` and any additional parameters that were
  previously passed through `model_builder_params`.【F:v_02/MainClass.py†L51-L122】【F:v_02/MainClass.py†L274-L282】

### v_03

- **Move into orchestrator**: Identical to v_02 plus shared handling of
  hyperparameters and evaluation.
- **Keep bespoke**:
  - `data` defaults now set `time_steps` to `3` to retain the short look-back
    window introduced in this version.【F:v_03/MainClass.py†L51-L59】
  - `callbacks` remain at patience `60` as per the legacy defaults.【F:v_03/MainClass.py†L62-L70】

### v_04

- **Move into orchestrator**: Same as v_03 (no extra bespoke pipeline code).
- **Keep bespoke**: Continue using the `time_steps=3` default and provide
  `block_configs`/`model_builder_params` through `config.model`.【F:v_04/MainClass.py†L50-L90】

### v_05

- **Move into orchestrator**: Same shared responsibilities.
- **Keep bespoke**: Same configuration as v_04 with `time_steps=3`.【F:v_05/MainClass.py†L50-L90】

### v_06

- **Move into orchestrator**: Shared handling of the training summary now covers
  the tweaked patience values.
- **Keep bespoke**:
  - Restore default patience via `callbacks={"early_stopping_patience": 30,
    "reduce_lr_patience": 1, "reduce_lr_factor": 0.1, "min_lr": 5e-7}`.【F:v_06/MainClass.py†L62-L71】
  - Maintain `time_steps=60` in `config.data` (same as v_01/v_02).

### v_07

- **Move into orchestrator**: Shared pipeline.
- **Keep bespoke**: Combine the v_03 time window (`time_steps=3`) with the
  patience defaults from v_01 (60 epochs).【F:v_07/MainClass.py†L51-L70】

### v_08

- **Move into orchestrator**: Shared pipeline, including repeated evaluation
  plots.
- **Keep bespoke**: Configure callbacks with `early_stopping_patience=20` (while
  other callback settings remain unchanged).【F:v_08/MainClass.py†L62-L70】

### v_09

- **Move into orchestrator**: Same as v_08.
- **Keep bespoke**: Identical callback settings (`early_stopping_patience=20`) and
  standard `time_steps=60` defaults.【F:v_09/MainClass.py†L54-L71】

### v_10

- **Move into orchestrator**: Shared pipeline features plus final export logic.
- **Keep bespoke**: Same configuration as v_09 (patience=20, `time_steps=60`).【F:v_10/MainClass.py†L50-L90】

## Suggested migration workflow

1. Register each version’s factories using `DependencyFactoryRegistry` *or*
   pass lambdas/partials directly through `OrchestratorDependencies`.
2. Build an `OrchestratorConfig` that mirrors the version’s constructor
   defaults (see notes above).
3. Replace the legacy `TimeSeriesModel` with a thin adapter that instantiates
   `TimeSeriesOrchestrator` and delegates to `run()`.
4. Incrementally move any version-specific logging or diagnostics into
   `OrchestratorHooks` so the shared core remains framework-agnostic.

This staged approach preserves historical behaviour while eliminating the
copy-paste maintenance burden across versions.
