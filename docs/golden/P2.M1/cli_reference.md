# Canonical CLI reference

The unified runner keeps the historical flag names used by the legacy scripts
while translating them onto the nested configuration keys expected by the
orchestrator.  `parse_initial_args` still recognises global utilities such as
`--config`, `--set`, and precision overrides, while any `--flag` tokens that
follow are resolved through the shared schema.

## Quick-start examples

```bash
# Inspect the resolved configuration without launching training
python -m bin.run_experiment --config v_05/default_config.yaml --print-config

# Override epochs and the residual attention dimensions for a V3+ config
python -m bin.run_experiment --config v_03/default_config.yaml \
  --epochs 120 --num_heads_residual_block 4 --key_dim_residual_block 6

# Enable the intermediate dense layer introduced in V5
python -m bin.run_experiment --config v_05/default_config.yaml \
  --use_intermediate_dense --intermediate_dense_units 512
```

Global options (`--config`, `--set`, `--seed`, `--mixed-precision`,
`--no-mixed-precision`, `--precision-policy`, `--file-path`, `--base-dir`,
`--print-config`) are parsed before any schema-driven flags and can be combined
freely with the legacy aliases.【F:bin/run_experiment.py†L147-L159】

## Flag catalogue

The table below lists the most frequently used legacy flags and the canonical
paths they affect.  The complete mapping—including Mixture-of-Experts and
residual-block specialists—is captured in `src/core/versioning/cli_schema.json`.

| Flag | Canonical path | Description |
| --- | --- | --- |
| `--data_file` | `file_path` | Replace the dataset path declared in the config file. |
| `--output_dir` | `base_dir` | Redirect run artefacts to a different directory. |
| `--time_steps` | `data.time_steps` | Adjust the sliding window length for the DataLoader. |
| `--epochs` | `training.epochs` | Override the number of training epochs. |
| `--batch_size` | `training.batch_size` | Override the mini-batch size. |
| `--optimizer_lr` | `model.optimizer_lr` | Tune the optimiser learning rate. |
| `--lstm_units` | `model.lstm_units` | Control the width of the BiLSTM stack. |
| `--recurrent_dropout_lstm` | `model.recurrent_dropout_lstm` | Enable/adjust recurrent dropout within the LSTM. |
| `--conv_l2_reg` | `model.conv_l2_reg` | Configure convolutional weight decay for residual-attention variants. |
| `--leaky_alpha_conv1` | `model.leaky_relu_alpha_conv1` | Adjust the LeakyReLU slope in the first convolution of a residual block. |

Each entry in the schema also includes a `type` descriptor to aid validation and
scripting.【F:src/core/versioning/cli_schema.json†L1-L95】  When a canonical path
uses a modern name (e.g. `model.leaky_relu_alpha_conv1`), the normalisation layer
mirrors the value onto the legacy alias (`model.leaky_relu_alpha_conv_1`) so the
version-specific builders receive the parameter they expect.【F:src/core/versioning/configuration.py†L1-L188】

## Toggles

Toggle-style flags set Boolean switches without requiring `=true/false` syntax.

| Flag | Canonical path | Effect |
| --- | --- | --- |
| `--disable_mixed_precision` | `runtime.mixed_precision.enabled` | Force full `float32` execution. |
| `--enable_mixed_precision` | `runtime.mixed_precision.enabled` | Opt-in to TensorFlow mixed precision policies. |
| `--use_intermediate_dense` | `model.use_intermediate_dense` | Enable the V5 intermediate dense layer. |

Supplying `--flag=value` for toggles halts translation so the runner can warn
about unexpected CLI usage.【F:bin/run_experiment.py†L109-L144】  Chaining
toggles with other overrides is safe because they are converted into
`--set`-style assignments behind the scenes.【F:bin/run_experiment.py†L118-L144】
