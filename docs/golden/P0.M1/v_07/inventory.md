# v_07 – Asset Audit

## Checksum Catalogue

| File | SHA256 |
| --- | --- |
| DataLoader.py | `8666e89bbb41360503fcbc856b7d426ab595d8b7b31193d4c277bda65650cb2c` |
| Evaluator.py | `b64aa8a85e962323a9cb3e80489548a5d4e53c3d4eec6ae8fffb83540a9c9554` |
| HistoryManager.py | `37f139d127fa3a703d3a617de1b481fafa96339e38234e6c65b60b0ba1e7f3ee` |
| MainClass.py | `9b3f49fdf9b827096bd64efafcdfcfe337199302483def066941b0a2d1fb152b` |
| ModelBuilder.py | `a4307b34b29cab363ad03d1e660ff31d352d9cbb75f54bd8813225fe61e76ee3` |
| ModelManager.py | `cb61d26e43565307c633ccd519f315f242e32f1abe5b0745cb054155b6a5f7a6` |
| Run.py | `73463200103eef722ed11cad5b0c836a73998eab26e104ac1dd9c3354011201b` |
| Trainer.py | `ed74d69a749d6f1a27d26da930343a0bdcb46b7f91c517b41e64adeb60a0d51f` |
| VisualPredictions.py | `7336a65a256f687cb87b16083ca5b7ec745e1e65d0f7344fcabcadc41ec89f63` |
| default_config.yaml | `659bc8d17457b785da29acfabf897fea33855b0d9335d71bb8fd585afa0ff9da` |
| dependencies.py | `d39f767f668516d87721ad244088e9363c4bda6e7f4e7831360e58afff703c4f` |

## Script Inventory

| Component | Role | Notes |
| --- | --- | --- |
| DataLoader.py | Shim around src.data.loader.ChronologicalDataLoader to preserve legacy signature. | Forwards to ChronologicalDataLoader with legacy defaults and optional override passthrough. |
| Evaluator.py | Generates forecasts, computes regression metrics, and orchestrates plotting. | Calculates metrics (R2, RMSE) and produces comparison plots via Matplotlib. |
| HistoryManager.py | Persists training history, hyperparameters, and callback outputs to disk. | Serialises history JSON, R2 traces, and metadata per run directory. |
| MainClass.py | High-level pipeline orchestrator tying together data loading, model building, training, and evaluation. | TimeSeriesModel orchestrates DataLoader → ModelBuilder → Trainer → Evaluator flow with logging. |
| ModelBuilder.py | Constructs the TensorFlow architecture specific to this legacy release. | Residual Conv1D stack plus post-LSTM MultiHeadAttention feeding an active Mix-of-Experts gating block before output. |
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
