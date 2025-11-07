"""Export experiment statistics to structured tabular formats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

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


def _summarise_tables(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for name, table in tables.items():
        numeric = table.select_dtypes(include="number")
        for column in numeric.columns:
            series = numeric[column]
            records.append(
                {
                    "source": name,
                    "metric": column,
                    "count": int(series.count()),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=1)) if series.count() > 1 else 0.0,
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
            )
    return pd.DataFrame.from_records(records)


def export_tables(metrics_paths: Iterable[Path], output_dir: Path) -> Dict[str, Path]:
    paths = [Path(p) for p in metrics_paths]
    tables = {path.stem: _load_table(path) for path in paths}

    summary = _summarise_tables(tables)
    combined = []
    for name, table in tables.items():
        copy = table.copy()
        copy.insert(0, "source", name)
        combined.append(copy)
    combined_df = pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "summary_csv": output_dir / "summary.csv",
        "summary_parquet": output_dir / "summary.parquet",
        "metrics_csv": output_dir / "metrics.csv",
        "metrics_parquet": output_dir / "metrics.parquet",
    }
    summary.to_csv(outputs["summary_csv"], index=False)
    summary.to_parquet(outputs["summary_parquet"], index=False)
    combined_df.to_csv(outputs["metrics_csv"], index=False)
    combined_df.to_parquet(outputs["metrics_parquet"], index=False)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serialise metrics to CSV and Parquet")
    parser.add_argument(
        "--metrics", nargs="+", type=Path, help="Input metrics files (CSV/JSON/Parquet)"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tables"))
    return parser


def main(argv: List[str] | None = None) -> None:  # pragma: no cover - CLI wrapper
    parser = build_parser()
    args = parser.parse_args(argv)
    outputs = export_tables(args.metrics, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
