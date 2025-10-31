"""Helpers that keep legacy entry points compatible with the shared runner."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List


def delegate_to_runner(config_path: Path, argv: Iterable[str]) -> int:
    """Invoke the central ``run_experiment`` entry point with a config.

    ``config_path`` is resolved relative to the legacy script while ``argv``
    contains any additional command line arguments the user supplied.
    """
    repo_root = config_path.resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from bin import run_experiment  # Import after mutating ``sys.path``

    argument_list: List[str] = ["--config", str(config_path)]
    argument_list.extend(argv)
    return run_experiment.main(argument_list)

