# v_06 – Asset Audit

## Checksum Catalogue

| File | SHA256 |
| --- | --- |
| DataLoader.py | `8666e89bbb41360503fcbc856b7d426ab595d8b7b31193d4c277bda65650cb2c` |
| Evaluator.py | `f93f00df0a5254c0ca25e2ac62bebce1a5f359d9bd96f0721ff066f741d09a0b` |
| HistoryManager.py | `37f139d127fa3a703d3a617de1b481fafa96339e38234e6c65b60b0ba1e7f3ee` |
| MainClass.py | `6825611eeaea227d2e1ba24db9364a38e8409a7fc0bb596a16e1cea81c7f1975` |
| ModelBuilder.py | `00319f81995e5c3fed3bd8ac29e3e611b74d30058bb793fa2e5bbfb81677c682` |
| ModelManager.py | `effc49041d65f2733ab4cab225718c704d78c99d07e55513923c23293abb1038` |
| Run.py | `73463200103eef722ed11cad5b0c836a73998eab26e104ac1dd9c3354011201b` |
| Trainer.py | `ed74d69a749d6f1a27d26da930343a0bdcb46b7f91c517b41e64adeb60a0d51f` |
| VisualPredictions.py | `7336a65a256f687cb87b16083ca5b7ec745e1e65d0f7344fcabcadc41ec89f63` |
| default_config.yaml | `570936427c1e392aa011dcbdbd353103c34b15db781466a3d1b6a08196b75b22` |
| dependencies.py | `fdec1c36ea1a9e9459d249d6e2c381f4e6989870b7c800c00065b3095c305b25` |

## Script Inventory

| Component | Role | Notes |
| --- | --- | --- |
| DataLoader.py | Shim around src.data.loader.ChronologicalDataLoader to preserve legacy signature. | Forwards to ChronologicalDataLoader with legacy defaults and optional override passthrough. |
| Evaluator.py | Generates forecasts, computes regression metrics, and orchestrates plotting. | Calculates metrics (R2, RMSE) and produces comparison plots via Matplotlib. |
| HistoryManager.py | Persists training history, hyperparameters, and callback outputs to disk. | Serialises history JSON, R2 traces, and metadata per run directory. |
| MainClass.py | High-level pipeline orchestrator tying together data loading, model building, training, and evaluation. | TimeSeriesModel orchestrates DataLoader → ModelBuilder → Trainer → Evaluator flow with logging. |
| ModelBuilder.py | Constructs the TensorFlow architecture specific to this legacy release. | Residual Conv1D stack carried over while introducing (but not wiring) a Mix-of-Experts layer for future use. |
| ModelManager.py | Coordinates model serialization across TensorFlow formats. | Handles saving models and scalers to filesystem for reproducibility. |
| Run.py | Entrypoint script that wires configuration, dependency registration, and pipeline execution. | Loads default_config, resolves factories, and launches TimeSeriesModel pipeline. |
| Trainer.py | Runs the Keras training loop with callbacks for LR scheduling, early stopping, and artifact logging. | Implements Keras callbacks (EpochTimerCallback, EarlyStopping, ReduceLROnPlateau, history writers). |
| VisualPredictions.py | Utilities for plotting predictions versus ground truth. | Provides helper functions to visualise predictions vs actual series. |
| default_config.yaml | Reference configuration consumed by Run.py and CLI tooling. | Tracks default hyperparameters, including split ratios and optimizer settings. |
| dependencies.py | Registers dependency factories with the shared orchestrator infrastructure. | Wraps src.core.versioning.create_dependency_bundle to expose factories and hooks. |

## Dependency Notes

- `dependencies.py` registers factories via `create_dependency_bundle` ensuring orchestrator isolation.
- Shared modules (`DataLoader`, `Trainer`, `MainClass`) import from `src.*` namespaces—no external packages beyond TensorFlow stack.
- Default configs pin run directories and mixed-precision toggles; confirm overrides when invoking CLI.
