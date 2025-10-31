# P0.M1 Version Inventory and Component Divergence

## Overview
This log captures the Python entry points distributed across the ten legacy experiment directories (`v_01`–`v_10`). Every version exposes the same top-level modules:

- `Run.py`
- `MainClass.py`
- `DataLoader.py`
- `ModelBuilder.py`
- `Trainer.py`
- `Evaluator.py`
- `ModelManager.py`
- `HistoryManager.py`
- `VisualPredictions.py`

The sections below document per-version checksums, note the dependency footprint, and highlight how the core pipeline components (DataLoader, ModelBuilder, Trainer) diverge.

## Per-Version Checksums
The following SHA-256 hashes cover every top-level script in each version directory. Any change to a module will surface as a checksum delta, enabling regression detection before refactors begin.

### v_01
```text
d3714938023d64494d184c82baa61b34f23d59e1514050192b4295225dbb1273  v_01/DataLoader.py
b64aa8a85e962323a9cb3e80489548a5d4e53c3d4eec6ae8fffb83540a9c9554  v_01/Evaluator.py
37f139d127fa3a703d3a617de1b481fafa96339e38234e6c65b60b0ba1e7f3ee  v_01/HistoryManager.py
9a666f30c6f68aa4f4698d388df686f3c2351e061f04194666a634ee0e849076  v_01/MainClass.py
cf6d61b2ca300596e8392f4664581aa19a32fb5bcb643326bdf8edb49fe1e1d8  v_01/ModelBuilder.py
effc49041d65f2733ab4cab225718c704d78c99d07e55513923c23293abb1038  v_01/ModelManager.py
382861aa38333c66e58c94df0c3e591b0ab72c5ab2fb5cfaf0f64c0a32a816df  v_01/Run.py
ed74d69a749d6f1a27d26da930343a0bdcb46b7f91c517b41e64adeb60a0d51f  v_01/Trainer.py
7336a65a256f687cb87b16083ca5b7ec745e1e65d0f7344fcabcadc41ec89f63  v_01/VisualPredictions.py
```

### v_02
```text
d3714938023d64494d184c82baa61b34f23d59e1514050192b4295225dbb1273  v_02/DataLoader.py
b64aa8a85e962323a9cb3e80489548a5d4e53c3d4eec6ae8fffb83540a9c9554  v_02/Evaluator.py
37f139d127fa3a703d3a617de1b481fafa96339e38234e6c65b60b0ba1e7f3ee  v_02/HistoryManager.py
5f813f41bb5915d976dc95d8d29886546b3c3e3923a50e50d3b45de7b88705b3  v_02/MainClass.py
5e503d3c77dbe452a34c8c14d031a638640631e46bcf88a9029ee87257b37a2e  v_02/ModelBuilder.py
effc49041d65f2733ab4cab225718c704d78c99d07e55513923c23293abb1038  v_02/ModelManager.py
1b364c164e28832c38585ac35bb67b0496014bcc225e5984796f5a1aa1da891f  v_02/Run.py
ed74d69a749d6f1a27d26da930343a0bdcb46b7f91c517b41e64adeb60a0d51f  v_02/Trainer.py
7336a65a256f687cb87b16083ca5b7ec745e1e65d0f7344fcabcadc41ec89f63  v_02/VisualPredictions.py
```

### v_03
```text
d3714938023d64494d184c82baa61b34f23d59e1514050192b4295225dbb1273  v_03/DataLoader.py
b64aa8a85e962323a9cb3e80489548a5d4e53c3d4eec6ae8fffb83540a9c9554  v_03/Evaluator.py
37f139d127fa3a703d3a617de1b481fafa96339e38234e6c65b60b0ba1e7f3ee  v_03/HistoryManager.py
9b3f49fdf9b827096bd64efafcdfcfe337199302483def066941b0a2d1fb152b  v_03/MainClass.py
9c91f47a42994ccb978728d07026401feaace8306feb01063940678eee9d28d5  v_03/ModelBuilder.py
effc49041d65f2733ab4cab225718c704d78c99d07e55513923c23293abb1038  v_03/ModelManager.py
5a31d40d4abb32e04a7a5db1cbb4a7690f2516b1d6a3479059dbdfae861ba8a5  v_03/Run.py
ed74d69a749d6f1a27d26da930343a0bdcb46b7f91c517b41e64adeb60a0d51f  v_03/Trainer.py
7336a65a256f687cb87b16083ca5b7ec745e1e65d0f7344fcabcadc41ec89f63  v_03/VisualPredictions.py
```

### v_04
```text
d3714938023d64494d184c82baa61b34f23d59e1514050192b4295225dbb1273  v_04/DataLoader.py
b64aa8a85e962323a9cb3e80489548a5d4e53c3d4eec6ae8fffb83540a9c9554  v_04/Evaluator.py
37f139d127fa3a703d3a617de1b481fafa96339e38234e6c65b60b0ba1e7f3ee  v_04/HistoryManager.py
9b3f49fdf9b827096bd64efafcdfcfe337199302483def066941b0a2d1fb152b  v_04/MainClass.py
4c9866a3f8ec01c1fb862ad1670142ff9eaf844f26130424cd5f21432d993fa7  v_04/ModelBuilder.py
effc49041d65f2733ab4cab225718c704d78c99d07e55513923c23293abb1038  v_04/ModelManager.py
060f63fb2d0a945f35c9aab231ff4020bf1917f16957cee081419b5048c58ff4  v_04/Run.py
ed74d69a749d6f1a27d26da930343a0bdcb46b7f91c517b41e64adeb60a0d51f  v_04/Trainer.py
7336a65a256f687cb87b16083ca5b7ec745e1e65d0f7344fcabcadc41ec89f63  v_04/VisualPredictions.py
```

### v_05
```text
d3714938023d64494d184c82baa61b34f23d59e1514050192b4295225dbb1273  v_05/DataLoader.py
b64aa8a85e962323a9cb3e80489548a5d4e53c3d4eec6ae8fffb83540a9c9554  v_05/Evaluator.py
37f139d127fa3a703d3a617de1b481fafa96339e38234e6c65b60b0ba1e7f3ee  v_05/HistoryManager.py
9b3f49fdf9b827096bd64efafcdfcfe337199302483def066941b0a2d1fb152b  v_05/MainClass.py
94a0521a1e1502f03db5439ac0ce6f073a0d3e616fdb0b0bb488b5ae54d66f8a  v_05/ModelBuilder.py
effc49041d65f2733ab4cab225718c704d78c99d07e55513923c23293abb1038  v_05/ModelManager.py
b2fc8cc658169fda2331fb61c1c18db72cea2ca85f6301535ee79f559586971f  v_05/Run.py
ed74d69a749d6f1a27d26da930343a0bdcb46b7f91c517b41e64adeb60a0d51f  v_05/Trainer.py
7336a65a256f687cb87b16083ca5b7ec745e1e65d0f7344fcabcadc41ec89f63  v_05/VisualPredictions.py
```

### v_06
```text
d3714938023d64494d184c82baa61b34f23d59e1514050192b4295225dbb1273  v_06/DataLoader.py
f93f00df0a5254c0ca25e2ac62bebce1a5f359d9bd96f0721ff066f741d09a0b  v_06/Evaluator.py
37f139d127fa3a703d3a617de1b481fafa96339e38234e6c65b60b0ba1e7f3ee  v_06/HistoryManager.py
6825611eeaea227d2e1ba24db9364a38e8409a7fc0bb596a16e1cea81c7f1975  v_06/MainClass.py
00319f81995e5c3fed3bd8ac29e3e611b74d30058bb793fa2e5bbfb81677c682  v_06/ModelBuilder.py
effc49041d65f2733ab4cab225718c704d78c99d07e55513923c23293abb1038  v_06/ModelManager.py
353a2e966a2e30f68893a3f2de259b2b108de40de405b998054d0303f1215b53  v_06/Run.py
ed74d69a749d6f1a27d26da930343a0bdcb46b7f91c517b41e64adeb60a0d51f  v_06/Trainer.py
7336a65a256f687cb87b16083ca5b7ec745e1e65d0f7344fcabcadc41ec89f63  v_06/VisualPredictions.py
```

### v_07
```text
d3714938023d64494d184c82baa61b34f23d59e1514050192b4295225dbb1273  v_07/DataLoader.py
b64aa8a85e962323a9cb3e80489548a5d4e53c3d4eec6ae8fffb83540a9c9554  v_07/Evaluator.py
37f139d127fa3a703d3a617de1b481fafa96339e38234e6c65b60b0ba1e7f3ee  v_07/HistoryManager.py
9b3f49fdf9b827096bd64efafcdfcfe337199302483def066941b0a2d1fb152b  v_07/MainClass.py
a4307b34b29cab363ad03d1e660ff31d352d9cbb75f54bd8813225fe61e76ee3  v_07/ModelBuilder.py
cb61d26e43565307c633ccd519f315f242e32f1abe5b0745cb054155b6a5f7a6  v_07/ModelManager.py
f19a93dad45b09725cccc98191a8833bbbdb8becfe5cca545efdc1fc6876ecf4  v_07/Run.py
ed74d69a749d6f1a27d26da930343a0bdcb46b7f91c517b41e64adeb60a0d51f  v_07/Trainer.py
7336a65a256f687cb87b16083ca5b7ec745e1e65d0f7344fcabcadc41ec89f63  v_07/VisualPredictions.py
```

### v_08
```text
d3714938023d64494d184c82baa61b34f23d59e1514050192b4295225dbb1273  v_08/DataLoader.py
158011e6a5befe6d6c20e45a188b42cb818751e873776f63e81a79268a642425  v_08/Evaluator.py
37f139d127fa3a703d3a617de1b481fafa96339e38234e6c65b60b0ba1e7f3ee  v_08/HistoryManager.py
d1c8550976e30f4286c9d261226a587a0322522168868e3fe9a311cc1de796b7  v_08/MainClass.py
5c0049ba4a7f563846554ceb2efdb51ba5fb150bfc26c21e5a35b067f289b95e  v_08/ModelBuilder.py
effc49041d65f2733ab4cab225718c704d78c99d07e55513923c23293abb1038  v_08/ModelManager.py
6e4ad7348662e2387109d61fa0d151cc2f14968f8278024c4a520b8dfc7b9118  v_08/Run.py
ed74d69a749d6f1a27d26da930343a0bdcb46b7f91c517b41e64adeb60a0d51f  v_08/Trainer.py
7336a65a256f687cb87b16083ca5b7ec745e1e65d0f7344fcabcadc41ec89f63  v_08/VisualPredictions.py
```

### v_09
```text
d3714938023d64494d184c82baa61b34f23d59e1514050192b4295225dbb1273  v_09/DataLoader.py
f93f00df0a5254c0ca25e2ac62bebce1a5f359d9bd96f0721ff066f741d09a0b  v_09/Evaluator.py
37f139d127fa3a703d3a617de1b481fafa96339e38234e6c65b60b0ba1e7f3ee  v_09/HistoryManager.py
d1c8550976e30f4286c9d261226a587a0322522168868e3fe9a311cc1de796b7  v_09/MainClass.py
3db0d1b2486afe9323e7d392719d47efcd2d668c525cb1b487a8a09b41a8004b  v_09/ModelBuilder.py
effc49041d65f2733ab4cab225718c704d78c99d07e55513923c23293abb1038  v_09/ModelManager.py
a68c24c9d17eba6effc4f5b380634796b5d360de727c5ef2cecbd02b453259d4  v_09/Run.py
ed74d69a749d6f1a27d26da930343a0bdcb46b7f91c517b41e64adeb60a0d51f  v_09/Trainer.py
7336a65a256f687cb87b16083ca5b7ec745e1e65d0f7344fcabcadc41ec89f63  v_09/VisualPredictions.py
```

### v_10
```text
d3714938023d64494d184c82baa61b34f23d59e1514050192b4295225dbb1273  v_10/DataLoader.py
f93f00df0a5254c0ca25e2ac62bebce1a5f359d9bd96f0721ff066f741d09a0b  v_10/Evaluator.py
37f139d127fa3a703d3a617de1b481fafa96339e38234e6c65b60b0ba1e7f3ee  v_10/HistoryManager.py
d1c8550976e30f4286c9d261226a587a0322522168868e3fe9a311cc1de796b7  v_10/MainClass.py
d86abb8a3bffa5763502d770d329224d93c972618adca55fcd1fa28baa844cda  v_10/ModelBuilder.py
effc49041d65f2733ab4cab225718c704d78c99d07e55513923c23293abb1038  v_10/ModelManager.py
ce940ca890da0fcc5beacf25d11ae1c082a9a726233e1485a3674ff4fe7a2954  v_10/Run.py
ed74d69a749d6f1a27d26da930343a0bdcb46b7f91c517b41e64adeb60a0d51f  v_10/Trainer.py
7336a65a256f687cb87b16083ca5b7ec745e1e65d0f7344fcabcadc41ec89f63  v_10/VisualPredictions.py
```

## Dependency Footprint
All versions rely on the same third-party stack, surfaced via import analysis of the shared modules:

| Library | Usage Notes |
| --- | --- |
| **TensorFlow / Keras** | Core deep-learning stack across `ModelBuilder.py`, `Trainer.py`, `Run.py`, `MainClass.py`, `Evaluator.py`, `HistoryManager.py`, and `ModelManager.py`. MultiHeadAttention, AdditiveAttention, AdamW, mixed precision, and callback APIs are exercised across versions.【F:v_01/Run.py†L1-L73】【F:v_08/ModelBuilder.py†L1-L74】 |
| **NumPy** | Base numerical operations in `DataLoader.py`, `Trainer.py`, `Evaluator.py`, `ModelManager.py`, and `MainClass.py` (e.g., argmin, array reshaping).【F:v_01/DataLoader.py†L1-L119】【F:v_01/Trainer.py†L1-L63】 |
| **pandas** | Tabular IO and plotting support in `DataLoader.py`, `ModelManager.py`, and `VisualPredictions.py` when assembling CSV-driven workflows.【F:v_01/DataLoader.py†L1-L71】【F:v_01/ModelManager.py†L1-L44】 |
| **scikit-learn** | `StandardScaler` in `DataLoader.py` and `r2_score` plus regression metrics in `Trainer.py`/`Evaluator.py`.【F:v_01/DataLoader.py†L1-L49】【F:v_01/Evaluator.py†L1-L65】 |
| **matplotlib** | Visualization callbacks and history plotting in `Trainer.py`, `Evaluator.py`, `HistoryManager.py`, and `VisualPredictions.py`.【F:v_01/Trainer.py†L1-L63】【F:v_01/HistoryManager.py†L1-L59】 |

No version introduces a divergent dependency stack; checksum parity for `DataLoader.py` and `Trainer.py` confirms identical copies across `v_01`–`v_10`.

## Component Divergence Highlights

- **DataLoader** – Identical implementation across all versions (`sha256 d3714938…`). Shared responsibilities: CSV ingestion, NaN drop, StandardScaler wrapping, sequence generation, and train/val/test splits.
- **Trainer** – Identical implementation across all versions (`sha256 ed74d69a…`). Manages callbacks (early stopping, ReduceLROnPlateau, epoch timing, custom history writers) and relies on matplotlib plus scikit-learn metrics.
- **ModelBuilder** – Only component evolving between releases. Key architectural changes:
  - `v_01`: Two Conv1D blocks feeding a BiLSTM block followed by Keras `Attention`; optional pooling is disabled by default.【F:v_01/ModelBuilder.py†L1-L121】
  - `v_02`: Switches to configurable convolutional block lists with embedded `MultiHeadAttention`; retains BiLSTM + final `Attention` stage.【F:v_02/ModelBuilder.py†L1-L130】
  - `v_03`: Adds residual Add connections around Conv1D + MHA pairs and introduces post-residual LeakyReLU plus configurable BatchNorm usage.【F:v_03/ModelBuilder.py†L1-L78】
  - `v_04`: Replaces the terminal attention with `AdditiveAttention` while keeping residual Conv1D + MHA stacks.【F:v_04/ModelBuilder.py†L1-L69】
  - `v_05`: Maintains residual stacks but routes LSTM output through another `MultiHeadAttention` head and optional intermediate Dense layer before regression output.【F:v_05/ModelBuilder.py†L1-L74】
  - `v_06`: Aligns with V8 hyperparameters, keeps residual Conv1D + MHA stacks, and defines a Mix-of-Experts layer for compatibility (not used in forward pass).【F:v_06/ModelBuilder.py†L1-L74】
  - `v_07`: Actively employs Mix-of-Experts after post-LSTM MHA, with dual pooling inside residual blocks and shortcut alignment rules.【F:v_07/ModelBuilder.py†L90-L181】
  - `v_08`: Codifies the Mix-of-Experts path and uses ReLU/LeakyReLU combinations; attention after LSTM uses high head/key dimensions (12 heads, key dim 50).【F:v_08/ModelBuilder.py†L1-L80】
  - `v_09`: Residual blocks adopt `AdditiveAttention`, revert to Additive attention after LSTM, and sustain Mix-of-Experts usage with batch normalization controls.【F:v_09/ModelBuilder.py†L1-L79】
  - `v_10`: Hybrid design combining MultiHeadAttention residuals with final `AdditiveAttention`, Mix-of-Experts, and tuned head counts (3 heads).【F:v_10/ModelBuilder.py†L1-L79】

These observations capture the primary architectural divergence points that future consolidation work must reconcile.
