"""Deterministic seeding helpers and reproducibility utilities."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch RNGs and configure CUDA flags."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def snapshot_torch_determinism() -> dict[str, Any]:
    """Capture the current torch/CUDA/cuDNN determinism settings."""

    snapshot: dict[str, Any] = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_enabled": torch.backends.cudnn.enabled,
    }

    if torch.cuda.is_available():
        snapshot.update(
            {
                "cuda_available": True,
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_seed_all": True,
            }
        )
    else:
        snapshot.update({"cuda_available": False, "cuda_device_count": 0})

    cudnn_version = None
    try:
        cudnn_version = torch.backends.cudnn.version()
    except AttributeError:  # pragma: no cover - defensive when cudnn is absent
        cudnn_version = None
    snapshot["cudnn_version"] = cudnn_version

    return snapshot


def get_deterministic_flags() -> dict[str, Any]:
    """Backward compatible alias for determinism snapshots."""

    return snapshot_torch_determinism()


def hash_config(cfg: DictConfig | Mapping[str, Any]) -> str:
    """Compute a stable SHA256 hash of a resolved Hydra/OmegaConf config."""

    if isinstance(cfg, DictConfig):
        container: Mapping[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    else:
        container = cfg

    serialised = json.dumps(container, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    """Compute a SHA256 digest for the provided file path."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_revision(default: str = "unknown") -> str:
    """Return the current Git commit SHA if available."""

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return default


def get_hardware_snapshot() -> dict[str, Any]:
    """Capture a lightweight snapshot of the host hardware."""

    snapshot: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }

    if torch.cuda.is_available():
        snapshot["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    else:
        snapshot["cuda_devices"] = []

    return snapshot


def build_run_provenance(
    seed: int,
    base_metadata: Mapping[str, Any] | None = None,
    *,
    dataset: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a provenance blob describing a seeded training run."""

    payload: MutableMapping[str, Any] = dict(base_metadata or {})
    payload["seed"] = seed
    payload.setdefault("git_sha", get_git_revision())
    payload.setdefault("hardware", get_hardware_snapshot())
    payload["deterministic_flags"] = snapshot_torch_determinism()

    torch_snapshot = {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mps_available": getattr(torch.backends, "mps", None) is not None
        and getattr(torch.backends.mps, "is_available", lambda: False)(),
    }
    try:
        torch_snapshot["git_version"] = torch.version.git_version
    except AttributeError:  # pragma: no cover - depends on torch build details
        torch_snapshot["git_version"] = None
    try:
        torch_snapshot["cudnn_enabled"] = torch.backends.cudnn.enabled
        torch_snapshot["cudnn_version"] = torch.backends.cudnn.version()
    except AttributeError:  # pragma: no cover - defensive when cudnn is absent
        torch_snapshot["cudnn_enabled"] = None
        torch_snapshot["cudnn_version"] = None

    payload["torch"] = torch_snapshot

    if dataset is not None:
        payload["dataset"] = dict(dataset)
    if extra is not None:
        payload.update(dict(extra))

    return dict(payload)
