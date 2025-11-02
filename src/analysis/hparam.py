"""Visualisation utilities for hyper-parameter search results."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _resolve_parameter_columns(
    results: pd.DataFrame, parameter_columns: Sequence[str] | None, metric_column: str
) -> list[str]:
    if parameter_columns:
        return [col for col in parameter_columns if col in results.columns]
    excluded = {metric_column, "trial"}
    return [col for col in results.columns if col not in excluded]


def plot_response_curves(
    results: pd.DataFrame,
    *,
    metric_column: str,
    output_dir: Path | str,
    parameter_columns: Sequence[str] | None = None,
) -> Dict[str, Dict[str, Path]]:
    """Render scatter/partial dependence plots and summaries for each tuned parameter."""

    if results.empty:
        LOGGER.warning("No results supplied for plotting")
        return {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metric_values = results[metric_column].to_numpy()
    artefacts: Dict[str, Dict[str, Path]] = {}

    for column in _resolve_parameter_columns(results, parameter_columns, metric_column):
        series = results[column]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(series, metric_values, alpha=0.7, edgecolor="none", label="trials")

        summary_df: pd.DataFrame
        if pd.api.types.is_numeric_dtype(series):
            sorted_df = results[[column, metric_column]].sort_values(column)
            if sorted_df[column].nunique() > 1:
                x_values = sorted_df[column].to_numpy()
                y_values = sorted_df[metric_column].rolling(window=3, min_periods=1).mean().to_numpy()
                ax.plot(x_values, y_values, color="tab:orange", linewidth=2, label="partial mean")
                summary_df = pd.DataFrame({column: x_values, "partial_mean": y_values})
            else:
                summary_df = sorted_df.rename(columns={metric_column: "partial_mean"})
        else:
            grouped = results.groupby(column)[metric_column].mean()
            positions = np.arange(len(grouped))
            ax.plot(positions, grouped.to_numpy(), marker="o", color="tab:orange", label="category mean")
            ax.set_xticks(positions)
            ax.set_xticklabels([str(val) for val in grouped.index], rotation=30, ha="right")
            summary_df = grouped.reset_index().rename(columns={metric_column: "partial_mean"})

        ax.set_xlabel(column)
        ax.set_ylabel(metric_column)
        ax.set_title(f"Response curve for {column}")
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.3)

        plot_path = output_path / f"{column.replace('.', '_')}_response.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        summary_path = output_path / f"{column.replace('.', '_')}_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        artefacts[column] = {"plot": plot_path, "summary": summary_path}

    return artefacts


__all__ = ["plot_response_curves"]

