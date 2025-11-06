"""CLI for generating calibration diagnostics from probabilistic forecasts."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics.calibration import (
    CalibrationSummary,
    crps_ensemble,
    interval_coverage,
    interval_coverage_error,
    pit_values,
    reliability_curve,
)

LOGGER = logging.getLogger(__name__)


def _normalise_path(path: Path) -> Path:
    return path if path.is_absolute() else path.resolve()


def _discover_prediction_file(root: Path) -> Path:
    if root.is_file():
        return root
    candidates = [
        *(root.glob("predictions.*")),
        *(root.glob("*.csv")),
        *(root.glob("*.parquet")),
    ]
    for candidate in candidates:
        if candidate.suffix.lower() in {".csv", ".parquet"}:
            return candidate
    raise FileNotFoundError(f"No predictions file found under {root}")


def _load_predictions(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension for {path}")
    if "y_true" not in frame.columns:
        raise KeyError("Predictions file must include a `y_true` column")
    return frame


def _extract_samples(frame: pd.DataFrame) -> np.ndarray:
    sample_cols = [col for col in frame.columns if col.startswith("sample_")]
    if not sample_cols:
        raise KeyError("Predictions must include columns named `sample_*` with ensemble draws")
    return frame[sample_cols].to_numpy(dtype=float)


def _extract_quantiles(frame: pd.DataFrame) -> Mapping[float, np.ndarray]:
    quantiles: dict[float, np.ndarray] = {}
    for column in frame.columns:
        if column.startswith("quantile_"):
            level_text = column.split("quantile_")[1]
            try:
                level = float(level_text)
            except ValueError:
                LOGGER.debug("Skipping quantile column %s", column)
                continue
            quantiles[level] = frame[column].to_numpy(dtype=float)
    return quantiles


def _quantile(samples: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    try:
        return np.quantile(samples, levels, axis=1, method="linear")
    except TypeError:  # pragma: no cover - compatibility for older NumPy
        return np.quantile(samples, levels, axis=1, interpolation="linear")


def _ensure_quantiles(
    quantiles: Mapping[float, np.ndarray],
    samples: np.ndarray,
    levels: Sequence[float],
) -> Mapping[float, np.ndarray]:
    available = dict(quantiles)
    missing_levels = [level for level in levels if level not in available]
    if missing_levels:
        LOGGER.info("Computing missing quantiles %s from samples", missing_levels)
        computed = _quantile(samples, missing_levels)
        for idx, level in enumerate(missing_levels):
            available[level] = computed[idx]
    return dict(sorted(available.items()))


def _central_interval_bounds(
    quantiles: Mapping[float, np.ndarray],
    confidence: float,
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = (1.0 - confidence) / 2.0
    lower_level = alpha
    upper_level = 1.0 - alpha
    if lower_level in quantiles and upper_level in quantiles:
        return quantiles[lower_level], quantiles[upper_level]
    LOGGER.info(
        "Interval %.2f not directly available; estimating from samples",
        confidence,
    )
    lower, upper = _quantile(samples, [lower_level, upper_level])
    return lower, upper


def _sanitise_label(text: str) -> str:
    allowed = []
    previous_sep = False
    for char in text:
        if char.isalnum():
            allowed.append(char.lower())
            previous_sep = False
        else:
            if not previous_sep:
                allowed.append("_")
                previous_sep = True
    label = "".join(allowed).strip("_")
    return label or "run"


def _pit_plot(values: np.ndarray, bins: int, destination: Path) -> None:
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    widths = np.diff(edges)
    centres = edges[:-1] + widths / 2
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(centres, counts / counts.sum(), width=widths, align="center", edgecolor="black")
    ax.axhline(1 / bins, color="red", linestyle="--", label="ideal")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Frequency")
    ax.set_title("Probability integral transform")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _coverage_plot(
    quantile_summary: CalibrationSummary,
    interval_points: Sequence[tuple[float, float]],
    destination: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", label="ideal")
    ax.plot(
        quantile_summary.nominal,
        quantile_summary.observed,
        marker="o",
        label="quantile",
    )
    if interval_points:
        nominal, observed = zip(*interval_points)
        ax.scatter(nominal, observed, color="C1", label="interval", zorder=3)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_title("Calibration reliability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _summarise_run(
    label: str,
    frame: pd.DataFrame,
    *,
    quantile_levels: Sequence[float],
    interval_levels: Sequence[float],
    pit_bins: int,
    output_dir: Path,
) -> list[dict[str, object]]:
    y_true = frame["y_true"].to_numpy(dtype=float)
    samples = _extract_samples(frame)
    quantiles = _extract_quantiles(frame)
    quantiles = _ensure_quantiles(quantiles, samples, quantile_levels)

    scores = crps_ensemble(y_true, samples)
    pit = pit_values(y_true, samples)
    summary_records: list[dict[str, object]] = [
        {"run": label, "metric": "crps", "level": "mean", "value": float(np.mean(scores))}
    ]

    interval_points: list[tuple[float, float]] = []
    for level in interval_levels:
        lower, upper = _central_interval_bounds(quantiles, level, samples)
        coverage = interval_coverage(y_true, lower, upper)
        interval_points.append((level, coverage))
        summary_records.append({"run": label, "metric": "coverage", "level": level, "value": coverage})
        summary_records.append(
            {
                "run": label,
                "metric": "coverage_error",
                "level": level,
                "value": interval_coverage_error(y_true, lower, upper, level),
            }
        )

    reliability = reliability_curve(y_true, quantiles)
    for nominal, observed in zip(reliability.nominal, reliability.observed):
        summary_records.append({"run": label, "metric": "reliability", "level": nominal, "value": observed})

    figs_dir = output_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    _pit_plot(pit, pit_bins, figs_dir / f"pit_hist_{label}.png")
    _coverage_plot(reliability, interval_points, figs_dir / f"coverage_{label}.png")
    return summary_records


def _stack_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True)


def generate_calibration(
    inputs: Sequence[Path],
    *,
    output_root: Path,
    quantiles: Sequence[float],
    intervals: Sequence[float],
    pit_bins: int,
) -> Path:
    records: list[dict[str, object]] = []
    run_frames: list[pd.DataFrame] = []
    labels: list[str] = []

    for path in inputs:
        resolved = _normalise_path(path)
        predictions_file = _discover_prediction_file(resolved)
        frame = _load_predictions(predictions_file)
        if resolved.is_file():
            parent_name = resolved.parent.name or "run"
            label_source = f"{parent_name}_{resolved.stem}"
        else:
            label_source = resolved.name
        label = _sanitise_label(label_source)
        run_frames.append(frame)
        labels.append(label)
        records.extend(
            _summarise_run(
                label,
                frame,
                quantile_levels=quantiles,
                interval_levels=intervals,
                pit_bins=pit_bins,
                output_dir=output_root,
            )
        )

    if run_frames:
        combined = _stack_frames(run_frames)
        records.extend(
            _summarise_run(
                "overall",
                combined,
                quantile_levels=quantiles,
                interval_levels=intervals,
                pit_bins=pit_bins,
                output_dir=output_root,
            )
        )
    else:
        raise ValueError("No prediction files provided")

    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "calibration.csv"
    frame = pd.DataFrame.from_records(records)
    frame.sort_values(["run", "metric", "level"], inplace=True, ignore_index=True)
    frame.to_csv(destination, index=False)
    LOGGER.info("Wrote calibration summary to %s", destination)
    return destination


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate calibration diagnostics")
    parser.add_argument("inputs", nargs="+", type=Path, help="Prediction files or directories")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("paper_outputs"),
        help="Directory where calibration outputs will be written",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=(0.1, 0.25, 0.5, 0.75, 0.9),
        help="Quantile levels to evaluate",
    )
    parser.add_argument(
        "--intervals",
        type=float,
        nargs="+",
        default=(0.5, 0.8, 0.95),
        help="Central interval coverages to evaluate",
    )
    parser.add_argument(
        "--pit-bins",
        type=int,
        default=10,
        help="Number of bins for PIT histograms",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    generate_calibration(
        [Path(path) for path in args.inputs],
        output_root=args.output_root,
        quantiles=args.quantiles,
        intervals=args.intervals,
        pit_bins=args.pit_bins,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
