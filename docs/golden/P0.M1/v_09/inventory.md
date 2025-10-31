# v_09 – Asset Audit

## Checksum Catalogue

| File | SHA256 |
| --- | --- |
| DataLoader.py | `8666e89bbb41360503fcbc856b7d426ab595d8b7b31193d4c277bda65650cb2c` |
| Evaluator.py | `f93f00df0a5254c0ca25e2ac62bebce1a5f359d9bd96f0721ff066f741d09a0b` |
| HistoryManager.py | `37f139d127fa3a703d3a617de1b481fafa96339e38234e6c65b60b0ba1e7f3ee` |
| MainClass.py | `d1c8550976e30f4286c9d261226a587a0322522168868e3fe9a311cc1de796b7` |
| ModelBuilder.py | `3db0d1b2486afe9323e7d392719d47efcd2d668c525cb1b487a8a09b41a8004b` |
| ModelManager.py | `effc49041d65f2733ab4cab225718c704d78c99d07e55513923c23293abb1038` |
| Run.py | `73463200103eef722ed11cad5b0c836a73998eab26e104ac1dd9c3354011201b` |
| Trainer.py | `ed74d69a749d6f1a27d26da930343a0bdcb46b7f91c517b41e64adeb60a0d51f` |
| VisualPredictions.py | `7336a65a256f687cb87b16083ca5b7ec745e1e65d0f7344fcabcadc41ec89f63` |
| default_config.yaml | `2124e4c105ed02e7b4ae312f52d24d556a5b1f6e94d88c6727214c4b11f6d55e` |
| dependencies.py | `231bb59fba38c2fb2b471f5e760415ae31cc1089dd3ac887b1feb89dc5df1b80` |

## Script Inventory

| Component | Role | Notes |
| --- | --- | --- |
| DataLoader.py | Shim around src.data.loader.ChronologicalDataLoader to preserve legacy signature. | Forwards to ChronologicalDataLoader with legacy defaults and optional override passthrough. |
| Evaluator.py | Generates forecasts, computes regression metrics, and orchestrates plotting. | Calculates metrics (R2, RMSE) and produces comparison plots via Matplotlib. |
| HistoryManager.py | Persists training history, hyperparameters, and callback outputs to disk. | Serialises history JSON, R2 traces, and metadata per run directory. |
| MainClass.py | High-level pipeline orchestrator tying together data loading, model building, training, and evaluation. | TimeSeriesModel orchestrates DataLoader → ModelBuilder → Trainer → Evaluator flow with logging. |
| ModelBuilder.py | Constructs the TensorFlow architecture specific to this legacy release. | AdditiveAttention replaces multi-head aggregation prior to the Mix-of-Experts block to emphasise positional weighting. |
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
