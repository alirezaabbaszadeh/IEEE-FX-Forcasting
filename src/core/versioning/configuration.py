"""Canonical schema definitions and normalisation utilities for legacy configs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple

CLI_SCHEMA_PATH = Path(__file__).with_name("cli_schema.json")

with CLI_SCHEMA_PATH.open(encoding="utf-8") as handle:
    LEGACY_CLI_SCHEMA: Dict[str, Any] = json.load(handle)


# ---------------------------------------------------------------------------
# Canonical field definitions
# ---------------------------------------------------------------------------

CANONICAL_TOP_LEVEL_SECTIONS = {
    "file_path",
    "base_dir",
    "data",
    "model",
    "training",
    "callbacks",
    "metadata",
    "dependencies",
    "runtime",
    "cli",
    "version",
}

CANONICAL_DATA_KEYS = {"time_steps", "train_ratio", "val_ratio", "test_ratio"}
CANONICAL_TRAINING_KEYS = {"epochs", "batch_size"}
CANONICAL_CALLBACK_KEYS = {
    "early_stopping_patience",
    "reduce_lr_patience",
    "reduce_lr_factor",
    "min_lr",
}
CANONICAL_METADATA_KEYS = {"legacy_version", "default_output_dir_name", "config_version"}
BLOCK_CONFIG_KEYS = {"filters", "kernel_size", "pool_size"}

MODEL_ALIAS_GROUPS: Dict[str, Tuple[str, ...]] = {
    "leaky_relu_alpha_conv1": ("leaky_relu_alpha_conv1", "leaky_relu_alpha_conv_1"),
    "leaky_relu_alpha_conv2": ("leaky_relu_alpha_conv2", "leaky_relu_alpha_conv_2"),
    "key_dim_residual_block": ("key_dim_residual_block", "key_dim_res_block"),
    "num_heads_residual_block": ("num_heads_residual_block", "num_heads_res_block"),
}

# All canonical model keys observed across legacy configs (after alias collapse)
CANONICAL_MODEL_KEYS = {
    "attention_after_lstm_heads",
    "attention_after_lstm_key_dim",
    "block_configs",
    "conv1_l2_reg",
    "conv2_l2_reg",
    "conv_l2_reg",
    "conv_l2_reg_res_block",
    "dense_l2_reg_after_flatten",
    "dense_units_after_flatten",
    "dense_units_before_output",
    "dropout_rate",
    "dropout_rate_before_output",
    "filters_conv1",
    "filters_conv2",
    "intermediate_dense_l2_reg",
    "intermediate_dense_units",
    "kernel_size_conv1",
    "kernel_size_conv2",
    "key_dim",
    "key_dim_conv_block",
    "key_dim_final_mha",
    "key_dim_residual_block",
    "l2_reg",
    "leaky_relu_alpha_after_add",
    "leaky_relu_alpha_after_add_res_block",
    "leaky_relu_alpha_after_residual_add",
    "leaky_relu_alpha_conv1",
    "leaky_relu_alpha_conv1_res_block",
    "leaky_relu_alpha_conv2",
    "leaky_relu_alpha_conv2_res_block",
    "leaky_relu_alpha_dense",
    "leaky_relu_alpha_dense_after_flatten",
    "leaky_relu_alpha_intermediate_dense",
    "leaky_relu_alpha_res_block",
    "leaky_relu_alpha_res_block2",
    "leaky_relu_alpha_res_block_1",
    "leaky_relu_alpha_res_block_2",
    "lstm_l2_reg",
    "lstm_units",
    "moe_leaky_relu_alpha",
    "moe_num_experts",
    "moe_units",
    "num_bilstm_layers",
    "num_heads",
    "num_heads_conv_block",
    "num_heads_final_mha",
    "num_heads_residual_block",
    "num_lstm_layers",
    "optimizer_clipnorm",
    "optimizer_clipvalue",
    "optimizer_lr",
    "optimizer_weight_decay",
    "output_activation",
    "output_l2_reg",
    "pool_size_conv1",
    "pool_size_conv2",
    "recurrent_dropout_lstm",
    "use_batchnorm_after_attention",
    "use_batchnorm_after_final_attention",
    "use_batchnorm_after_final_mha",
    "use_batchnorm_after_lstm",
    "use_batchnorm_after_moe",
    "use_batchnorm_after_post_lstm_mha",
    "use_batchnorm_intermediate_dense",
    "use_dropout_before_output",
    "use_intermediate_dense",
    "use_pooling_conv1",
    "use_pooling_conv2",
}


@dataclass
class NormalisationReport:
    """Outcome of the legacy configuration normalisation."""

    payload: Dict[str, Any]
    warnings: List[str]


def _ensure_mapping(payload: Mapping[str, Any], key: str) -> MutableMapping[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"Section '{key}' must be a mapping (got {type(value)!r})")
    return dict(value)


def _synchronise_aliases(model_section: MutableMapping[str, Any], warnings: List[str]) -> None:
    for canonical, variants in MODEL_ALIAS_GROUPS.items():
        present_values = [model_section[name] for name in variants if name in model_section]
        if canonical in model_section:
            canonical_value = model_section[canonical]
        elif present_values:
            canonical_value = present_values[0]
            model_section[canonical] = canonical_value
        else:
            continue
        for name in variants:
            if name == canonical:
                continue
            existing = model_section.get(name)
            if existing is not None and existing != canonical_value:
                warnings.append(
                    f"model.{name}: Conflicting value disagrees with canonical '{canonical}'"
                )
            model_section[name] = canonical_value


def canonicalise_model_key(name: str) -> str:
    for canonical, variants in MODEL_ALIAS_GROUPS.items():
        if name in variants:
            return canonical
    return name


def canonicalise_path(path: str) -> str:
    parts = path.split(".")
    if parts and parts[0] == "model" and len(parts) >= 2:
        parts[1] = canonicalise_model_key(parts[1])
    return ".".join(parts)


def merge_cli_schema(version_schema: Mapping[str, Any] | None) -> Dict[str, Any]:
    merged = json.loads(json.dumps(LEGACY_CLI_SCHEMA))
    if not version_schema:
        return merged
    for category in ("flags", "toggles"):
        custom_entries = version_schema.get(category, {})
        if not isinstance(custom_entries, Mapping):
            continue
        target = merged.setdefault(category, {})
        for name, spec in custom_entries.items():
            if not isinstance(spec, Mapping):
                continue
            normalised = dict(spec)
            path = normalised.get("path")
            if isinstance(path, str):
                normalised["path"] = canonicalise_path(path)
            target[name] = normalised
    return merged


def normalise_legacy_config(payload: Mapping[str, Any]) -> NormalisationReport:
    config: Dict[str, Any] = dict(payload)
    warnings: List[str] = []

    for key in ("data", "model", "training", "callbacks", "metadata"):
        config[key] = _ensure_mapping(config, key)

    data_section = config["data"]
    unknown_data = sorted(set(data_section) - CANONICAL_DATA_KEYS)
    for key in unknown_data:
        warnings.append(f"data.{key}: Unrecognised data field")

    training_section = config["training"]
    unknown_training = sorted(set(training_section) - CANONICAL_TRAINING_KEYS)
    for key in unknown_training:
        warnings.append(f"training.{key}: Unrecognised training field")

    callbacks_section = config["callbacks"]
    unknown_callbacks = sorted(set(callbacks_section) - CANONICAL_CALLBACK_KEYS)
    for key in unknown_callbacks:
        warnings.append(f"callbacks.{key}: Unrecognised callback field")

    metadata_section = config["metadata"]
    unknown_metadata = sorted(set(metadata_section) - CANONICAL_METADATA_KEYS)
    for key in unknown_metadata:
        warnings.append(f"metadata.{key}: Unrecognised metadata field")

    model_section = config["model"]
    _synchronise_aliases(model_section, warnings)
    canonical_keys = {canonicalise_model_key(name) for name in model_section}
    unknown_model = sorted(canonical_keys - CANONICAL_MODEL_KEYS)
    for key in unknown_model:
        warnings.append(f"model.{key}: Unrecognised model field")

    block_configs = model_section.get("block_configs")
    if isinstance(block_configs, list):
        for index, block in enumerate(block_configs):
            if not isinstance(block, Mapping):
                warnings.append(f"model.block_configs[{index}]: Expected mapping")
                continue
            unknown_block_keys = sorted(set(block) - BLOCK_CONFIG_KEYS)
            for key in unknown_block_keys:
                warnings.append(
                    f"model.block_configs[{index}].{key}: Unrecognised block field"
                )
    elif block_configs is not None:
        warnings.append("model.block_configs: Expected list")

    top_level_unknown = sorted(set(config) - CANONICAL_TOP_LEVEL_SECTIONS)
    for key in top_level_unknown:
        warnings.append(f"{key}: Unexpected top-level field")

    return NormalisationReport(payload=config, warnings=warnings)


def log_normalisation_warnings(logger: logging.Logger, warnings: Iterable[str]) -> None:
    for message in warnings:
        logger.warning("%s", message)


__all__ = [
    "LEGACY_CLI_SCHEMA",
    "NormalisationReport",
    "canonicalise_path",
    "log_normalisation_warnings",
    "merge_cli_schema",
    "normalise_legacy_config",
]
