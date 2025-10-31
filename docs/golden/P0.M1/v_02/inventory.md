# v_02 – Asset Audit

## Checksum Catalogue

| File | SHA256 |
| --- | --- |
| DataLoader.py | `8666e89bbb41360503fcbc856b7d426ab595d8b7b31193d4c277bda65650cb2c` |
| Evaluator.py | `b64aa8a85e962323a9cb3e80489548a5d4e53c3d4eec6ae8fffb83540a9c9554` |
| HistoryManager.py | `37f139d127fa3a703d3a617de1b481fafa96339e38234e6c65b60b0ba1e7f3ee` |
| MainClass.py | `5f813f41bb5915d976dc95d8d29886546b3c3e3923a50e50d3b45de7b88705b3` |
| ModelBuilder.py | `5e503d3c77dbe452a34c8c14d031a638640631e46bcf88a9029ee87257b37a2e` |
| ModelManager.py | `effc49041d65f2733ab4cab225718c704d78c99d07e55513923c23293abb1038` |
| Run.py | `73463200103eef722ed11cad5b0c836a73998eab26e104ac1dd9c3354011201b` |
| Trainer.py | `ed74d69a749d6f1a27d26da930343a0bdcb46b7f91c517b41e64adeb60a0d51f` |
| VisualPredictions.py | `7336a65a256f687cb87b16083ca5b7ec745e1e65d0f7344fcabcadc41ec89f63` |
| default_config.yaml | `4e044e0696ef1928b2f9aff71bf56540bd7636541edf46648b64ccaf8c9247ba` |
| dependencies.py | `ca87518f3abfb8e4f653b80380878959105463d1f384904c061ab7c0a6a16799` |

## Script Inventory

| Component | Role | Notes |
| --- | --- | --- |
| DataLoader.py | Shim around src.data.loader.ChronologicalDataLoader to preserve legacy signature. | Forwards to ChronologicalDataLoader with legacy defaults and optional override passthrough. |
| Evaluator.py | Generates forecasts, computes regression metrics, and orchestrates plotting. | Calculates metrics (R2, RMSE) and produces comparison plots via Matplotlib. |
| HistoryManager.py | Persists training history, hyperparameters, and callback outputs to disk. | Serialises history JSON, R2 traces, and metadata per run directory. |
| MainClass.py | High-level pipeline orchestrator tying together data loading, model building, training, and evaluation. | TimeSeriesModel orchestrates DataLoader → ModelBuilder → Trainer → Evaluator flow with logging. |
| ModelBuilder.py | Constructs the TensorFlow architecture specific to this legacy release. | Configurable Conv1D blocks wrap MultiHeadAttention layers before the BiLSTM and final attention head. |
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
