# Legacy configuration inventory

This inventory captures the shared structure and divergences present in the ten
`v_*` configuration files that feed the unified orchestrator.  It focuses on the
sections consumed by `OrchestratorConfig` and the additional metadata that the
runner must adapt when translating historical experiments.  All configuration
files expose the same top-level sections—`file_path`, `base_dir`, `data`,
`model`, `training`, `callbacks`, `metadata`, `dependencies`, `runtime`, and
`cli`—with section contents diverging per experiment variant.【F:v_01/default_config.yaml†L1-L126】【F:v_03/default_config.yaml†L20-L120】【F:v_06/default_config.yaml†L21-L120】【F:v_09/default_config.yaml†L21-L99】  The orchestrator consumes only the
`file_path`, `base_dir`, and the `data`, `model`, `training`, `callbacks`, and
`metadata` dictionaries, leaving the remaining blocks to the runner for runtime
configuration and dependency wiring.【F:src/core/orchestrator.py†L93-L103】

## Required sections and common keys

The following keys are present across **all** versions:

- `data`: `time_steps`, `train_ratio`, `val_ratio`, `test_ratio`【F:v_01/default_config.yaml†L5-L10】【F:v_03/default_config.yaml†L20-L23】【F:v_06/default_config.yaml†L21-L24】【F:v_09/default_config.yaml†L21-L24】
- `training`: `epochs`, `batch_size`【F:v_01/default_config.yaml†L11-L14】
- `callbacks`: `early_stopping_patience`, `reduce_lr_patience`, `reduce_lr_factor`, `min_lr`【F:v_01/default_config.yaml†L15-L20】
- `metadata`: `legacy_version`, `default_output_dir_name`【F:v_01/default_config.yaml†L50-L53】
- `model`: the shared baseline includes `block_configs`, `lstm_units`, `optimizer_lr`, and `optimizer_clipnorm` (the latter defaults to `null` where unused).【F:v_01/default_config.yaml†L21-L48】【F:v_03/default_config.yaml†L21-L48】【F:v_06/default_config.yaml†L21-L49】【F:v_09/default_config.yaml†L21-L44】

These common elements map directly into `OrchestratorConfig`, guaranteeing that
baseline preprocessing and training loops are always parameterised for the
shared runner.【F:src/core/orchestrator.py†L93-L103】

### `block_configs`

When present, `model.block_configs` is a list of dictionaries containing the
same trio of keys—`filters`, `kernel_size`, and `pool_size`—even though later
variants introduce attention heads or mixture-of-experts modules elsewhere in
the model section.【F:v_03/default_config.yaml†L22-L28】【F:v_06/default_config.yaml†L22-L28】【F:v_08/default_config.yaml†L22-L28】

## Per-version model extensions

Beyond the shared baseline, each variant layers on additional knobs.  The table
below lists the most salient `model`-level additions relative to the common key
set.

| Version | Notable additions |
| --- | --- |
| `v_01` | Convolutional hyperparameters (`filters_conv{1,2}`, `kernel_size_conv{1,2}`), regularisers (`conv{1,2}_l2_reg`, `output_l2_reg`), pooling controls, BiLSTM dropout/regularisation, and batch-normalisation toggles.【F:v_01/default_config.yaml†L21-L48】 |
| `v_02` | Residual convolution configuration via `conv_l2_reg`, attention block sizing (`key_dim_conv_block`, `num_heads_conv_block`), and renamed LeakyReLU parameters (`leaky_relu_alpha_conv_{1,2}`) that keep underscores for compatibility with the original builder signatures.【F:v_02/default_config.yaml†L21-L48】 |
| `v_03`–`v_05` | Residual attention stacks with `num_heads_residual_block`/`key_dim_residual_block`, additional dropout before output, and in `v_05` the optional intermediate dense layer (`use_intermediate_dense`, `intermediate_dense_units`, `leaky_relu_alpha_intermediate_dense`).【F:v_03/default_config.yaml†L21-L48】【F:v_05/default_config.yaml†L21-L48】 |
| `v_06`–`v_07` | Residual-block specific knobs (`*_res_block`), dense layers after flattening, and (in `v_07`) mixture-of-experts controls (`moe_num_experts`, `moe_units`, `moe_leaky_relu_alpha`).【F:v_06/default_config.yaml†L21-L49】【F:v_07/default_config.yaml†L21-L48】 |
| `v_08` | Post-LSTM attention heads (`attention_after_lstm_heads`, `attention_after_lstm_key_dim`), dense activations preceding the output, and mixture-of-experts sizing reused from `v_07`.【F:v_08/default_config.yaml†L21-L46】 |
| `v_09`–`v_10` | Simplified residual hyperparameters (`leaky_relu_alpha_res_block_{1,2}`, `leaky_relu_alpha_after_add`) and persistent mixture-of-experts settings shared with `v_07`/`v_08`.【F:v_09/default_config.yaml†L21-L44】【F:v_10/default_config.yaml†L21-L44】 |

The presence of both canonical and underscore-delimited parameter names (for
example `leaky_relu_alpha_conv1` versus `leaky_relu_alpha_conv_1`) reflects the
function signatures expected by the corresponding `ModelBuilder` classes.  The
runner must therefore preserve legacy aliases when normalising payloads so that
factory instantiation succeeds.

## Mismatches to account for in the runner

- **Top-level extras**: Every configuration retains `runtime`, `dependencies`,
  and `cli` blocks that the orchestrator ignores but the runner must process to
  configure seeds, factories, and CLI aliases.  These are stripped or merged
  before instantiating `OrchestratorConfig` so only the supported sections reach
  the orchestrator.【F:v_01/default_config.yaml†L54-L126】【F:src/core/orchestrator.py†L93-L103】
- **Alias proliferation**: Several variants ship both legacy and modernised
  parameter names (e.g. `key_dim_res_block` versus `key_dim_residual_block`),
  requiring the normalisation layer to mirror values across aliases so CLI
  overrides mapped to canonical keys also update the signatures consumed by the
  legacy builders.【F:v_03/default_config.yaml†L29-L48】【F:v_06/default_config.yaml†L29-L46】
- **Version-tagged metadata**: The `version` field is preserved as
  `metadata.config_version` to ensure provenance alongside the legacy output
  directory hints when the orchestrator serialises run summaries.【F:v_01/default_config.yaml†L2-L53】

## Impact on tooling

The shared CLI schema (`src/core/versioning/cli_schema.json`) now enumerates the
legacy flag names and points them at canonical nested paths so historic command
lines continue to work while new documentation presents consistent field names.【F:src/core/versioning/cli_schema.json†L1-L95】
The normalisation shim materialises both canonical and legacy aliases, logging
any unsupported keys for follow-up analysis.【F:src/core/versioning/configuration.py†L1-L188】
