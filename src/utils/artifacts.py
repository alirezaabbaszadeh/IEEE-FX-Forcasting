"""Utilities for managing artifact metadata and resolved configurations."""
from __future__ import annotations

import hashlib
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

from omegaconf import DictConfig, OmegaConf

from src.utils.repro import hash_config, hash_file


def resolve_config(cfg: DictConfig) -> Mapping[str, Any]:
    """Return a resolved, JSON-serialisable representation of a Hydra config."""

    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, Mapping):
        raise TypeError("Resolved config must be a mapping")
    return container


def ensure_config_snapshot(cfg: DictConfig, artifacts_root: Path) -> dict[str, Any]:
    """Persist the resolved config under ``artifacts_root/configs`` and return metadata."""

    resolved = resolve_config(cfg)
    config_hash = hash_config(resolved)

    config_dir = artifacts_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{config_hash}.yaml"

    if not config_path.exists():
        OmegaConf.save(config=OmegaConf.create(resolved), f=str(config_path))

    return {
        "hash": config_hash,
        "path": str(config_path),
    }


def collect_environment_lockfiles(project_root: Path) -> list[dict[str, str]]:
    """Capture hashes for known environment lockfiles located in the project root."""

    candidates = (
        "environment.yml",
        "requirements.txt",
        "requirements-dev.txt",
        "poetry.lock",
        "pyproject.toml",
    )
    records: list[dict[str, str]] = []
    for name in candidates:
        path = project_root / name
        if not path.exists():
            continue
        records.append({
            "path": str(path.relative_to(project_root)),
            "sha256": hash_file(path),
        })
    return records


def compute_dataset_checksums(dataset_metadata: Mapping[str, Any]) -> dict[str, str]:
    """Compute SHA256 digests for dataset index payloads."""

    keys = ("train_index", "val_index", "test_index")
    checksums: dict[str, str] = {}
    for key in keys:
        value = dataset_metadata.get(key)
        if value is None:
            continue
        if isinstance(value, Mapping):
            serialisable = dict(value)
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            serialisable = list(value)
        else:
            serialisable = value
        payload = json.dumps(serialisable, default=str, sort_keys=True)
        checksums[key] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return checksums


def build_run_metadata(
    metadata: Mapping[str, Any],
    *,
    seed: int | None = None,
    device: str | None = None,
    artifact_index: Mapping[str, Any] | None = None,
    dataset_checksums: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Normalise run metadata emitted alongside training artifacts."""

    payload: MutableMapping[str, Any] = {}

    if seed is not None:
        payload["seed"] = seed

    config_info = metadata.get("config")
    if isinstance(config_info, Mapping):
        payload["config"] = dict(config_info)
        if "config_hash" not in metadata and "hash" in config_info:
            payload["config_hash"] = config_info["hash"]

    if "config_hash" in metadata:
        payload["config_hash"] = metadata["config_hash"]

    for key in ("environment", "git_sha", "hardware", "deterministic_flags", "torch"):
        if key in metadata:
            payload[key] = metadata[key]

    resolved_device = device or metadata.get("device")
    if resolved_device is not None:
        payload["device"] = resolved_device

    dataset = metadata.get("dataset")
    if isinstance(dataset, Mapping):
        dataset_payload = dict(dataset)
        if dataset_checksums:
            dataset_payload["checksums"] = dict(dataset_checksums)
        payload["dataset"] = dataset_payload

    if artifact_index:
        payload["artifacts"] = dict(artifact_index)

    baseline_metrics = metadata.get("baseline_metrics")
    if isinstance(baseline_metrics, Mapping):
        payload["baseline_metrics"] = dict(baseline_metrics)

    return dict(payload)
