# v_01 – Asset Audit

## Checksum Catalogue

| File | SHA256 |
| --- | --- |
| DataLoader.py | `8666e89bbb41360503fcbc856b7d426ab595d8b7b31193d4c277bda65650cb2c` |
| Evaluator.py | `b64aa8a85e962323a9cb3e80489548a5d4e53c3d4eec6ae8fffb83540a9c9554` |
| HistoryManager.py | `37f139d127fa3a703d3a617de1b481fafa96339e38234e6c65b60b0ba1e7f3ee` |
| MainClass.py | `9a666f30c6f68aa4f4698d388df686f3c2351e061f04194666a634ee0e849076` |
| ModelBuilder.py | `cf6d61b2ca300596e8392f4664581aa19a32fb5bcb643326bdf8edb49fe1e1d8` |
| ModelManager.py | `effc49041d65f2733ab4cab225718c704d78c99d07e55513923c23293abb1038` |
| Run.py | `d63c01fc8a3dc865191a7e14898b4c49800b41444c73457f6d7f2d94ff71a43e` |
| Trainer.py | `ed74d69a749d6f1a27d26da930343a0bdcb46b7f91c517b41e64adeb60a0d51f` |
| VisualPredictions.py | `7336a65a256f687cb87b16083ca5b7ec745e1e65d0f7344fcabcadc41ec89f63` |
| default_config.yaml | `94ab9b3264651fdbe95ddfffdb7d6889393b7c453081a2c99f1df9e8c0aff56e` |
| dependencies.py | `6c8a988dc36635780f9140e34603ca436a6f6ec522c90ef27055559039014ac2` |

## Script Inventory

| Component | Role | Notes |
| --- | --- | --- |
| DataLoader.py | Shim around src.data.loader.ChronologicalDataLoader to preserve legacy signature. | Forwards to ChronologicalDataLoader with legacy defaults and optional override passthrough. |
| Evaluator.py | Generates forecasts, computes regression metrics, and orchestrates plotting. | Calculates metrics (R2, RMSE) and produces comparison plots via Matplotlib. |
| HistoryManager.py | Persists training history, hyperparameters, and callback outputs to disk. | Serialises history JSON, R2 traces, and metadata per run directory. |
| MainClass.py | High-level pipeline orchestrator tying together data loading, model building, training, and evaluation. | TimeSeriesModel orchestrates DataLoader → ModelBuilder → Trainer → Evaluator flow with logging. |
| ModelBuilder.py | Constructs the TensorFlow architecture specific to this legacy release. | Two Conv1D blocks feed a BiLSTM and single-head Attention head; defaults keep pooling and dropout disabled. |
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
