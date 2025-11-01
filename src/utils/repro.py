"""Deterministic seeding helpers and reproducibility utilities."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from typing import Any, Mapping

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


def get_deterministic_flags() -> dict[str, bool]:
    """Return the current deterministic flags for cuDNN usage."""

    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def hash_config(cfg: DictConfig | Mapping[str, Any]) -> str:
    """Compute a stable SHA256 hash of a resolved Hydra/OmegaConf config."""

    if isinstance(cfg, DictConfig):
        container: Mapping[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    else:
        container = cfg

    serialised = json.dumps(container, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


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
