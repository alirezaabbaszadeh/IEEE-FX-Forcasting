"""Rebuild publication tables and figures from stored training artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

from scripts.export_figures import export_figures
from scripts.export_tables import export_tables
from src.reporting.aggregates import collate_run_group, discover_run_roots


def _resolve_artifact_path(base: Path, entry: str) -> Path:
    path = Path(entry)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _discover_metadata_files(root: Path) -> List[Path]:
    runs_root = root / "runs"
    if not runs_root.exists():
        return []
    return sorted(runs_root.glob("**/metadata.json"))


def _collect_metrics_paths(metadata_files: Sequence[Path]) -> List[Path]:
    discovered: list[Path] = []
    for meta_path in metadata_files:
        payload = json.loads(meta_path.read_text())
        artifacts = payload.get("artifacts", {})
        metrics_entry = artifacts.get("metrics")
        if not metrics_entry:
            continue
        metrics_path = _resolve_artifact_path(meta_path.parent, str(metrics_entry))
        if metrics_path.exists():
            discovered.append(metrics_path)
    unique: list[Path] = []
    seen = set()
    for path in discovered:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _collect_config_paths(metadata_files: Sequence[Path], project_root: Path) -> List[Path]:
    configs: list[Path] = []
    seen = set()
    for meta_path in metadata_files:
        payload = json.loads(meta_path.read_text())
        config = payload.get("config")
        if not isinstance(config, dict):
            continue
        config_path = config.get("path")
        if not config_path:
            continue
        resolved = (project_root / config_path).resolve()
        if resolved.exists() and resolved not in seen:
            seen.add(resolved)
            configs.append(resolved)
    return configs


def _populate_aggregates(
    artifacts_root: Path,
    *,
    aggregates_dir: Path | None = None,
) -> dict[str, list[str]]:
    runs_root = artifacts_root / "runs"
    outputs: dict[str, list[str]] = {}
    for run_root in discover_run_roots(runs_root):
        produced = collate_run_group(run_root, aggregates_root=aggregates_dir)
        relative = run_root.relative_to(artifacts_root)
        outputs[str(relative)] = [str(path) for path in produced.values()]
    return outputs


def rebuild_publication_assets(
    *,
    artifacts_root: Path,
    tables_dir: Path,
    figures_dir: Path,
    project_root: Path,
    aggregates_dir: Path | None = None,
) -> dict[str, object]:
    aggregate_outputs = _populate_aggregates(artifacts_root, aggregates_dir=aggregates_dir)
    metadata_files = _discover_metadata_files(artifacts_root)
    metrics_paths = _collect_metrics_paths(metadata_files)
    config_paths = _collect_config_paths(metadata_files, project_root)

    if not metrics_paths:
        raise FileNotFoundError(
            "No metrics files discovered under artifacts/runs. Train the model before reproducing outputs."
        )

    table_outputs = export_tables(metrics_paths, tables_dir)
    figure_paths = export_figures(metrics_paths, figures_dir)

    return {
        "metrics_sources": [str(path) for path in metrics_paths],
        "tables": {name: str(path) for name, path in table_outputs.items()},
        "figures": [str(path) for path in figure_paths],
        "configs": [str(path.relative_to(project_root)) for path in config_paths],
        "aggregates": aggregate_outputs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild tables and figures from stored artifacts.")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts"),
        help="Root directory containing run artifacts",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        help="Output directory for regenerated tables",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        help="Output directory for regenerated figures",
    )
    parser.add_argument(
        "--aggregates-dir",
        type=Path,
        default=None,
        help="Optional directory for aggregate CSV outputs (defaults to artifacts/aggregates)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve stored config snapshots",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to write a JSON manifest summarising outputs",
    )
    parser.add_argument(
        "--populate-only",
        action="store_true",
        help="Populate aggregate CSV outputs without regenerating tables or figures",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:  # pragma: no cover - CLI wrapper
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    artifacts_root = args.artifacts_root.resolve()
    tables_dir = (args.tables_dir or artifacts_root / "tables").resolve()
    figures_dir = (args.figures_dir or artifacts_root / "figures").resolve()
    project_root = args.project_root.resolve()

    aggregates_dir = args.aggregates_dir.resolve() if args.aggregates_dir is not None else None

    if args.populate_only:
        aggregate_outputs = _populate_aggregates(artifacts_root, aggregates_dir=aggregates_dir)
        manifest = {"aggregates": aggregate_outputs}
    else:
        manifest = rebuild_publication_assets(
            artifacts_root=artifacts_root,
            tables_dir=tables_dir,
            figures_dir=figures_dir,
            project_root=project_root,
            aggregates_dir=aggregates_dir,
        )

    if args.manifest is not None:
        args.manifest.write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
