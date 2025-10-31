"""Compatibility shim that forwards legacy runs to the shared runner."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from src.core.legacy_adapter import delegate_to_runner

DEFAULT_CONFIG = Path(__file__).with_name("default_config.yaml")


def main(argv: Iterable[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    return delegate_to_runner(DEFAULT_CONFIG, args)


if __name__ == "__main__":
    sys.exit(main())
