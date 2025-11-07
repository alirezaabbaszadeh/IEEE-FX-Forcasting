"""Convert experiment metrics into publication-grade vector graphics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd


def _load_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".json", ".jsonl"}:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            return pd.DataFrame([payload])
        return pd.DataFrame(payload)
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported metrics format: {path}")


def _make_line_plot(df: pd.DataFrame, column: str, output: Path) -> None:
    if "epoch" in df.columns:
        x = df["epoch"].to_numpy()
        xlabel = "Epoch"
    else:
        x = range(len(df))
        xlabel = "Step"
    y = df[column].to_numpy()
    fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
    ax.plot(x, y, marker="o", linewidth=1.5)
    ax.set_title(column.replace("_", " ").title())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(column)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def export_figures(metrics_paths: Iterable[Path], output_dir: Path) -> List[Path]:
    paths = [Path(p) for p in metrics_paths]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: List[Path] = []
    for metrics_path in paths:
        table = _load_table(metrics_path)
        numeric_cols = table.select_dtypes(include="number").columns
        stem = metrics_path.stem
        for column in numeric_cols:
            figure_path = output_dir / f"{stem}_{column}.svg"
            _make_line_plot(table, column, figure_path)
            figure_paths.append(figure_path)
    return figure_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render metrics into vector figures.")
    parser.add_argument(
        "--metrics", nargs="+", type=Path, help="Input metrics files (CSV/JSON/Parquet)"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/figures"))
    return parser


def main(argv: List[str] | None = None) -> None:  # pragma: no cover - CLI wrapper
    parser = build_parser()
    args = parser.parse_args(argv)
    figure_paths = export_figures(args.metrics, args.output_dir)
    for path in figure_paths:
        print(path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
