"""Create the anonymous artifact archive for the TMLR submission."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence
import zipfile

_CANONICAL_TIMESTAMP = (2024, 1, 1, 0, 0, 0)
_DIR_MODE = 0o755
_FILE_MODE = 0o644
_DEFAULT_LIMIT = 100 * 1024 * 1024  # 100 MiB

_EXCLUDED_DIR_NAMES = {"__pycache__", ".mypy_cache", ".pytest_cache", ".git"}
_EXCLUDED_FILE_NAMES = {".DS_Store"}
_EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pyd", ".so", ".dylib"}


@dataclass(frozen=True)
class ManifestEntry:
    source: Path
    destination: Path


_MANIFEST: tuple[ManifestEntry, ...] = (
    ManifestEntry(Path("scripts"), Path("scripts")),
    ManifestEntry(Path("src"), Path("src")),
    ManifestEntry(Path("configs"), Path("configs")),
    ManifestEntry(Path("data/sample.csv"), Path("data/sample.csv")),
    ManifestEntry(Path("environment.yml"), Path("environment.yml")),
    ManifestEntry(Path("requirements-dev.txt"), Path("requirements-dev.txt")),
    ManifestEntry(Path("pyproject.toml"), Path("pyproject.toml")),
    ManifestEntry(Path("docs/tmlr/README_ANON.md"), Path("README_ANON.md")),
)


def _iter_tree(root: Path) -> Iterator[Path]:
    stack = [root]
    while stack:
        current = stack.pop()
        if current.is_dir():
            if current is not root and current.name in _EXCLUDED_DIR_NAMES:
                continue
            yield current
            children = sorted(current.iterdir(), key=lambda p: p.name)
            for child in reversed(children):
                stack.append(child)
        else:
            if current.name in _EXCLUDED_FILE_NAMES:
                continue
            if current.suffix in _EXCLUDED_SUFFIXES:
                continue
            yield current


def _iter_sources(project_root: Path, manifest: Sequence[ManifestEntry]) -> Iterator[tuple[Path, Path]]:
    for entry in manifest:
        source = (project_root / entry.source).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Manifest entry {entry.source} does not exist")
        destination_root = entry.destination
        if source.is_dir():
            for path in _iter_tree(source):
                relative = path.relative_to(source) if path != source else Path()
                arcname = destination_root / relative
                yield path, arcname
        else:
            yield source, destination_root


def _zip_info_from_path(path: Path, arcname: Path) -> zipfile.ZipInfo:
    zipped_name = arcname.as_posix()
    if path.is_dir() and not zipped_name.endswith("/"):
        zipped_name = f"{zipped_name}/"
    info = zipfile.ZipInfo(zipped_name)
    info.date_time = _CANONICAL_TIMESTAMP
    mode = _DIR_MODE if path.is_dir() else _FILE_MODE
    info.external_attr = (mode & 0xFFFF) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _add_path_to_zip(archive: zipfile.ZipFile, source: Path, arcname: Path) -> None:
    info = _zip_info_from_path(source, arcname)
    if source.is_dir():
        archive.writestr(info, b"")
        return
    data = source.read_bytes()
    archive.writestr(info, data)


def _validate_limit(archive_path: Path, limit_bytes: int) -> None:
    size = archive_path.stat().st_size
    if size > limit_bytes:
        archive_path.unlink(missing_ok=True)
        raise ValueError(
            f"Archive exceeds limit: {size / (1024 * 1024):.2f} MiB > {limit_bytes / (1024 * 1024):.2f} MiB"
        )


def build_archive(
    *,
    project_root: Path,
    output_path: Path,
    size_limit: int = _DEFAULT_LIMIT,
    manifest: Sequence[ManifestEntry] = _MANIFEST,
) -> None:
    if not project_root.is_dir():
        raise NotADirectoryError(project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    entries = list(_iter_sources(project_root, manifest))
    # Ensure directories appear before their children for deterministic archives.
    entries.sort(key=lambda item: (0 if item[0].is_dir() else 1, item[1].as_posix()))

    with zipfile.ZipFile(output_path, mode="w") as archive:
        for source, arcname in entries:
            _add_path_to_zip(archive, source, arcname)

    _validate_limit(output_path, size_limit)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the anonymous artifact archive.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifact_anonymous.zip"),
        help="Destination archive path (default: artifact_anonymous.zip)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root that hosts the manifest contents",
    )
    parser.add_argument(
        "--size-limit",
        type=int,
        default=_DEFAULT_LIMIT,
        help="Maximum archive size in bytes (default: 100 MiB)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    build_archive(
        project_root=args.project_root.resolve(),
        output_path=args.output.resolve(),
        size_limit=args.size_limit,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
