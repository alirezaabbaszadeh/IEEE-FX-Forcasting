"""Diebold-Mariano test utilities and command-line entry point."""
from __future__ import annotations

import argparse
import logging
import math
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _newey_west_variance(diff: np.ndarray, lag: int) -> float:
    diff_centered = diff - diff.mean()
    gamma_0 = float(np.dot(diff_centered, diff_centered) / diff.size)
    if lag <= 0:
        return gamma_0
    variance = gamma_0
    for h in range(1, min(lag, diff.size - 1) + 1):
        weight = 1 - h / (lag + 1)
        cov = float(np.dot(diff_centered[h:], diff_centered[:-h]) / diff.size)
        variance += 2 * weight * cov
    return max(variance, 1e-12)


def diebold_mariano(
    errors_a: Sequence[float],
    errors_b: Sequence[float],
    *,
    power: int = 2,
    use_newey_west: bool = False,
    lag: int = 1,
) -> Dict[str, float]:
    """Compute the Diebold-Mariano statistic for two forecast error series."""
    e_a = np.asarray(errors_a, dtype=float)
    e_b = np.asarray(errors_b, dtype=float)
    if e_a.shape != e_b.shape:
        raise ValueError("Error sequences must share the same length")

    if power == 2:
        losses_a = e_a**2
        losses_b = e_b**2
    else:
        losses_a = np.abs(e_a) ** power
        losses_b = np.abs(e_b) ** power

    diff = losses_a - losses_b
    mean_diff = diff.mean()
    n = diff.size
    if use_newey_west:
        variance = _newey_west_variance(diff, lag)
    else:
        variance = diff.var(ddof=1)
    denom = math.sqrt(variance / n + 1e-12)
    dm_stat = mean_diff / denom if denom else float("nan")
    p_value = 2 * NormalDist().cdf(-abs(dm_stat))
    return {
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "mean_diff": float(mean_diff),
        "variance": float(variance),
    }


def construct_dm_comparisons(
    repository: pd.DataFrame,
    *,
    pair_col: str = "pair",
    horizon_col: str = "horizon",
    model_col: str = "model",
    seed_col: str = "seed",
    value_col: str = "value",
    lag: int = 1,
    power: int = 2,
) -> pd.DataFrame:
    """Construct pairwise Diebold-Mariano comparisons from a repository DataFrame."""
    rows: list[Dict[str, object]] = []
    if repository.empty:
        return pd.DataFrame(rows)

    grouped = repository.groupby([pair_col, horizon_col], dropna=False)
    for (pair, horizon), group in grouped:
        pivot = group.pivot_table(index=seed_col, columns=model_col, values=value_col)
        pivot = pivot.dropna(axis=0, how="any")
        if pivot.empty:
            continue
        for model_a, model_b in combinations(pivot.columns, 2):
            values_a = pivot[model_a].to_numpy()
            values_b = pivot[model_b].to_numpy()
            dm = diebold_mariano(
                values_a,
                values_b,
                power=power,
                use_newey_west=True,
                lag=lag,
            )
            rows.append(
                {
                    pair_col: pair,
                    horizon_col: horizon,
                    "model_a": model_a,
                    "model_b": model_b,
                    **dm,
                }
            )
    return pd.DataFrame(rows)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Diebold-Mariano analysis from a DM cache")
    parser.add_argument("dm_cache", type=Path, help="Path to a DM cache CSV file")
    parser.add_argument("--baseline-model", required=True, help="Name of the baseline model")
    parser.add_argument("--metric", default="squared_error", help="Metric column in the DM cache")
    parser.add_argument("--run-id", default="paper", help="Identifier used for output folders")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_outputs"),
        help="Directory where paper-ready outputs are stored",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for tests")
    parser.add_argument(
        "--assumption-alpha",
        type=float,
        default=0.05,
        help="Significance level for assumption diagnostics",
    )
    parser.add_argument("--newey-west-lag", type=int, default=1, help="Lag parameter for Newey-West HAC")
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Treat larger metric values as better (default assumes losses)",
    )
    parser.add_argument("--random-seed", type=int, default=None, help="Seed for bootstrap reproducibility")
    parser.add_argument("--n-bootstrap", type=int, default=2000, help="Number of bootstrap samples")
    parser.add_argument(
        "--power",
        type=int,
        default=2,
        help="Loss power used when forming DM test statistics",
    )
    return parser.parse_args(argv)


def _write_dm_outputs(
    dm_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dm_table.to_csv(output_dir / "diebold_mariano.csv", index=False)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    dm_cache = pd.read_csv(args.dm_cache)

    from src.analysis.stats import analyze_dm_cache

    tables = analyze_dm_cache(
        dm_cache,
        run_id=str(args.run_id),
        output_dir=args.output_dir,
        baseline_model=args.baseline_model,
        metric=args.metric,
        alpha=args.alpha,
        assumption_alpha=args.assumption_alpha,
        newey_west_lag=args.newey_west_lag,
        higher_is_better=args.higher_is_better,
        random_state=args.random_seed,
        n_bootstrap=args.n_bootstrap,
    )

    dm_table = tables.get("diebold_mariano")
    if dm_table is None:
        LOGGER.warning("No Diebold-Mariano table produced from DM cache")
        dm_table = pd.DataFrame()
    if not dm_table.empty and args.power != 2:
        LOGGER.info("Recomputing DM statistics with power=%s", args.power)
        recomputed: list[pd.Series] = []
        for _, row in dm_table.iterrows():
            mask_a = (
                (dm_cache["pair"] == row["pair"]) &
                (dm_cache["horizon"] == row["horizon"]) &
                (dm_cache["model"] == row["model_a"])
            )
            mask_b = (
                (dm_cache["pair"] == row["pair"]) &
                (dm_cache["horizon"] == row["horizon"]) &
                (dm_cache["model"] == row["model_b"])
            )
            errors_a = dm_cache.loc[mask_a, args.metric].to_numpy(dtype=float)
            errors_b = dm_cache.loc[mask_b, args.metric].to_numpy(dtype=float)
            dm = diebold_mariano(
                errors_a,
                errors_b,
                power=args.power,
                use_newey_west=True,
                lag=args.newey_west_lag,
            )
            recomputed.append(pd.Series(dm))
        if recomputed:
            dm_table = dm_table.drop(columns=["dm_stat", "p_value", "mean_diff", "variance"], errors="ignore")
            dm_table = pd.concat([dm_table.reset_index(drop=True), pd.DataFrame(recomputed)], axis=1)

    _write_dm_outputs(dm_table, args.output_dir)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
