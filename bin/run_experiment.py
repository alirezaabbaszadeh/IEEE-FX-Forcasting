#!/usr/bin/env python3
"""Unified CLI for running legacy IEEE FX experiments via the shared orchestrator."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.orchestrator import (  # noqa: E402  # pylint: disable=wrong-import-position
    OrchestratorConfig,
    OrchestratorDependencies,
    TimeSeriesOrchestrator,
)


def _load_yaml_module() -> Optional[Any]:
    import importlib.util

    spec = importlib.util.find_spec("yaml")
    if spec is None:
        return None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None  # for mypy
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


_YAML = _load_yaml_module()


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if _YAML is not None:
        return _YAML.safe_load(text)  # type: ignore[no-any-return]
    return json.loads(text)


def parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        if raw.startswith("0x"):
            return int(raw, 16)
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            stripped = raw
            if (stripped.startswith("\"") and stripped.endswith("\"")) or (
                stripped.startswith("'") and stripped.endswith("'")
            ):
                stripped = stripped[1:-1]
            return stripped


def set_deep(mapping: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor: MutableMapping[str, Any] = mapping
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], MutableMapping):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def apply_overrides(mapping: MutableMapping[str, Any], overrides: Iterable[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override '{override}' must be formatted as key=value")
        key, value = override.split("=", 1)
        set_deep(mapping, key.strip(), parse_value(value.strip()))


def resolve_path(path_value: Optional[str], base_dir: Path) -> Optional[str]:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def translate_cli_arguments(arguments: List[str], schema: Optional[Mapping[str, Any]]) -> Tuple[List[str], List[str]]:
    if not arguments or schema is None:
        return [], arguments

    flag_specs: Mapping[str, Any] = schema.get("flags", {})  # type: ignore[arg-type]
    toggle_specs: Mapping[str, Any] = schema.get("toggles", {})  # type: ignore[arg-type]

    translated: List[str] = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if not token.startswith("--"):
            break
        name, eq, attached = token[2:].partition("=")
        if name in flag_specs:
            spec = flag_specs[name]
            if eq:
                value = attached
            else:
                index += 1
                if index >= len(arguments):
                    raise ValueError(f"Flag '--{name}' expects a value")
                value = arguments[index]
            translated.append(f"{spec['path']}={value}")
        elif name in toggle_specs:
            spec = toggle_specs[name]
            translated.append(f"{spec['path']}={'true' if spec['value'] else 'false'}")
            if eq:
                # Toggles should not use --flag=value syntax
                break
        else:
            break
        index += 1
    return translated, arguments[index:]


def parse_initial_args(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True, help="Path to the configuration YAML/JSON file")
    parser.add_argument("--set", action="append", default=[], help="Override configuration values (key=value)")
    parser.add_argument("--seed", type=int, help="Seed to apply across Python, NumPy, and TensorFlow")
    parser.add_argument("--mixed-precision", dest="mixed_precision", action="store_true")
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    parser.add_argument("--precision-policy", help="Mixed precision policy name (e.g. mixed_float16)")
    parser.add_argument("--file-path", help="Override the dataset file path")
    parser.add_argument("--base-dir", help="Override the orchestrator base directory")
    parser.add_argument("--print-config", action="store_true", help="Echo the resolved configuration and exit")
    parser.set_defaults(mixed_precision=None)
    return parser.parse_known_args(argv)


def configure_environment(seed: Optional[int], precision_cfg: Mapping[str, Any]) -> None:
    if seed is not None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
    enabled = precision_cfg.get("enabled", False)
    policy_name = precision_cfg.get("policy", "mixed_float16")
    if enabled:
        try:
            policy = mixed_precision.Policy(policy_name)
            mixed_precision.set_global_policy(policy)
            logging.getLogger(__name__).info(
                "Mixed precision enabled with policy '%s' (compute=%s variable=%s)",
                policy_name,
                policy.compute_dtype,
                policy.variable_dtype,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.getLogger(__name__).warning(
                "Failed to enable mixed precision policy '%s': %s", policy_name, exc, exc_info=True
            )
    else:
        mixed_precision.set_global_policy("float32")


def resolve_factory(reference: Any) -> Any:
    if callable(reference):
        return reference
    if not isinstance(reference, str):
        raise TypeError(f"Unsupported factory reference type: {type(reference)!r}")
    module_name, _, attr = reference.rpartition(".")
    if not module_name:
        raise ValueError(f"Factory reference '{reference}' must include a module path")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def build_orchestrator(
    config_payload: Mapping[str, Any],
    dependencies_payload: Mapping[str, Any],
    logger: logging.Logger,
) -> TimeSeriesOrchestrator:
    def _normalise_factory(reference: Any) -> Callable[..., Any]:
        if callable(reference):
            return reference
        return resolve_factory(reference)

    payload = dict(dependencies_payload)
    hooks_ref = payload.pop("hooks", None)
    hooks = None
    if hooks_ref is not None:
        hooks_factory = _normalise_factory(hooks_ref)
        hooks = hooks_factory()

    orchestrator_config = OrchestratorConfig(
        file_path=config_payload["file_path"],
        base_dir=config_payload.get("base_dir", "TimeSeries_Project_Runs/"),
        data=dict(config_payload.get("data", {})),
        model=dict(config_payload.get("model", {})),
        training=dict(config_payload.get("training", {})),
        callbacks=dict(config_payload.get("callbacks", {})),
        metadata=dict(config_payload.get("metadata", {})),
    )

    dependencies = OrchestratorDependencies(
        data_loader=_normalise_factory(payload["data_loader"]),
        model_builder=_normalise_factory(payload["model_builder"]),
        trainer=_normalise_factory(payload["trainer"]),
        evaluator=_normalise_factory(payload["evaluator"]),
        history_manager=_normalise_factory(payload["history_manager"]),
    )

    return TimeSeriesOrchestrator(
        config=orchestrator_config,
        dependencies=dependencies,
        hooks=hooks,
        logger=logger,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    argument_list = list(argv if argv is not None else sys.argv[1:])
    initial_args, remaining = parse_initial_args(argument_list)
    config_path = Path(initial_args.config).expanduser().resolve()
    config_data = load_config(config_path)

    cli_schema = config_data.get("cli")
    cli_overrides, leftover = translate_cli_arguments(remaining, cli_schema)
    if leftover:
        raise SystemExit(f"Unrecognised arguments: {' '.join(leftover)}")

    overrides = list(initial_args.set)
    overrides.extend(cli_overrides)

    if initial_args.file_path:
        overrides.append(f"file_path={initial_args.file_path}")
    if initial_args.base_dir:
        overrides.append(f"base_dir={initial_args.base_dir}")
    if initial_args.precision_policy:
        overrides.append(f"runtime.mixed_precision.policy={initial_args.precision_policy}")

    apply_overrides(config_data, overrides)

    config_dir = config_path.parent
    config_data["file_path"] = resolve_path(config_data.get("file_path"), config_dir)
    config_data["base_dir"] = resolve_path(config_data.get("base_dir"), config_dir)

    runtime_cfg = dict(config_data.pop("runtime", {}))
    config_data.pop("cli", None)
    dependencies_payload = dict(config_data.pop("dependencies", {}))

    metadata = dict(config_data.get("metadata", {}))
    version_tag = config_data.pop("version", None)
    if version_tag is not None:
        metadata.setdefault("config_version", version_tag)
    config_data["metadata"] = metadata

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("run_experiment")

    seed = initial_args.seed if initial_args.seed is not None else runtime_cfg.get("seed")
    if initial_args.mixed_precision is not None:
        runtime_cfg.setdefault("mixed_precision", {})
        runtime_cfg["mixed_precision"]["enabled"] = initial_args.mixed_precision
    precision_cfg = runtime_cfg.get("mixed_precision", {})
    configure_environment(seed, precision_cfg)

    orchestrator = build_orchestrator(config_data, dependencies_payload, logger)
    try:
        if initial_args.print_config:
            payload = {
                "config": config_data,
                "runtime": runtime_cfg,
                "dependencies": dependencies_payload,
            }
            json.dump(payload, sys.stdout, indent=2)
            sys.stdout.write("\n")
            return 0

        logger.info("Launching experiment with configuration: %s", json.dumps(config_data, indent=2))
        context = orchestrator.run()
        success = bool(context.get("success"))
        return 0 if success else 1
    finally:
        orchestrator.close()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
