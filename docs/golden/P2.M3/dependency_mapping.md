# Legacy dependency integration mapping

This milestone records how each legacy `MainClass` variant is now wired through the shared
`TimeSeriesOrchestrator`.  Every experiment version exposes a dedicated `dependencies.py`
module that translates its original constructor surfaces into factories compatible with the
orchestrator lifecycle.

## Factory coverage per version

All ten versions share the same adaptation pattern generated through
`src/core/versioning/legacy.py`.  The table below lists the concrete factories and hooks
exported by each version-specific module.

| Version | Data loader factory | Model builder factory | Trainer factory | Evaluator factory | Hooks |
|---------|---------------------|-----------------------|-----------------|-------------------|-------|
| v_01–v_10 | `create_data_loader` | `create_model_builder` | `create_trainer` | `create_evaluator` | `build_hooks` |

Each factory retains the legacy constructor parameters while injecting orchestrator context
where required:

* `create_data_loader` forwards the legacy CSV/path arguments and stores the loader on the
  orchestrator so callbacks that expect `self.data_loader` continue to operate.【F:src/core/versioning/legacy.py†L39-L77】
* `create_model_builder` and `create_trainer` reuse the legacy hyperparameter schema while
  binding the orchestrator instance as `main_model_instance`, preserving checkpointing and
  callback behaviours that assumed `TimeSeriesModel` ownership.【F:src/core/versioning/legacy.py†L79-L118】
* `create_evaluator` pulls split tensors and scalers from the orchestrator-managed data
  payload and ensures evaluation artefacts continue to land inside the run directory.【F:src/core/versioning/legacy.py†L120-L150】
* `create_history_manager` mirrors the one-argument constructor but keeps room for future
  keyword extensions through signature filtering.【F:src/core/versioning/legacy.py†L152-L161】

Each `v_*/dependencies.py` module surfaces the generated factories, a registration helper, and
the shared hook builder.【F:v_01/dependencies.py†L1-L36】【F:v_05/dependencies.py†L1-L36】  The
modules also expose registry key metadata so that downstream tooling can register factories if
string indirection becomes preferable.

## Hook behaviour

A lightweight hook bundle is emitted per version to bridge timing metadata from the legacy
trainer callbacks into the orchestrator’s summary logic.  After each training run the hook
copies the recorded epoch durations into `TimeSeriesOrchestrator.epoch_durations`, enabling the
core `_process_training_summary` helper to compute runtime aggregates that match the original
reports.【F:src/core/versioning/legacy.py†L163-L170】  Hooks are resolved during orchestrator
construction through the updated `bin/run_experiment.py` bootstrap, which now accepts factory
objects or import strings for dependency wiring and optional hook builders.【F:bin/run_experiment.py†L196-L229】

## Deviations and gaps

* The legacy implementations stored their summary payload under the key
  `training_summary_for_hyperparameters` with human-readable labels.  The orchestrator now
  emits a normalised dictionary (`best_val_loss_epoch_num`, `avg_time_per_epoch_seconds`, etc.).
  Because the hook reuses the orchestrator’s summariser, the field names differ from the
  original exports; this discrepancy is documented here for consumers that relied on the prior
  nomenclature.【F:src/core/orchestrator.py†L370-L420】
