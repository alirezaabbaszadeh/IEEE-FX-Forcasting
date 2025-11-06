"""Utilities for writing reproducibility manifests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from src.utils.repro import (
    get_git_revision,
    get_hardware_snapshot,
    snapshot_torch_determinism,
)


def _coerce_seed(value: Any) -> int | str | None:
    """Best-effort conversion of the provided value into a JSON serialisable seed."""

    if value is None:
        return None
    if isinstance(value, bool):  # bool is a subclass of int; keep explicit guard
        return int(value)
    if isinstance(value, (int, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, float):
        return int(value)
    return str(value)


def _normalise_lockfiles(metadata: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """Extract lockfile digests from the metadata payload."""

    if not metadata:
        return []
    environment = metadata.get("environment")
    if not isinstance(environment, Mapping):
        return []
    lockfiles = environment.get("lockfiles")
    normalised: list[dict[str, str]] = []
    if isinstance(lockfiles, list):
        for entry in lockfiles:
            if not isinstance(entry, Mapping):
                continue
            path = entry.get("path")
            digest = entry.get("sha256")
            if path is None or digest is None:
                continue
            normalised.append({"path": str(path), "sha256": str(digest)})
    return normalised


def _normalise_mapping(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return {str(key): value for key, value in payload.items()}


def build_manifest(metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a structured manifest describing the execution environment."""

    metadata = metadata or {}
    manifest: MutableMapping[str, Any] = {}

    seed_value = _coerce_seed(metadata.get("seed"))
    if seed_value is not None:
        manifest["seed"] = seed_value

    if "device" in metadata:
        manifest["device"] = metadata["device"]

    deterministic_flags = metadata.get("deterministic_flags")
    if not isinstance(deterministic_flags, Mapping):
        deterministic_flags = snapshot_torch_determinism()
    manifest["determinism"] = _normalise_mapping(deterministic_flags)

    hardware = metadata.get("hardware")
    if not isinstance(hardware, Mapping):
        hardware = get_hardware_snapshot()
    manifest["hardware"] = _normalise_mapping(hardware)

    torch_info = metadata.get("torch")
    if isinstance(torch_info, Mapping):
        torch_payload = _normalise_mapping(torch_info)
    else:
        try:  # pragma: no cover - defensive when torch is unavailable
            import torch  # type: ignore
        except ModuleNotFoundError:  # pragma: no cover - runtime fallback
            torch_payload = {}
        else:  # pragma: no cover - exercised when torch available
            torch_payload = {
                "version": torch.__version__,
                "cuda_version": getattr(torch.version, "cuda", None),
                "git_version": getattr(torch.version, "git_version", None),
            }
            try:
                torch_payload["cudnn_version"] = torch.backends.cudnn.version()
                torch_payload["cudnn_enabled"] = torch.backends.cudnn.enabled
            except AttributeError:
                torch_payload["cudnn_version"] = None
                torch_payload["cudnn_enabled"] = None
    manifest["libraries"] = {"torch": torch_payload}

    git_sha = metadata.get("git_sha")
    if not git_sha:
        git_sha = get_git_revision()
    manifest["git"] = {"sha": str(git_sha)}

    lockfiles = _normalise_lockfiles(metadata)
    manifest["environment"] = {"lockfiles": lockfiles}

    return dict(manifest)


def write_manifest(path: Path, metadata: Mapping[str, Any] | None = None) -> Path:
    """Persist a reproducibility manifest to ``path``."""

    manifest = build_manifest(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return path


__all__ = ["build_manifest", "write_manifest"]
