"""Model confidence set utilities and CLI entry point."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .spa import superior_predictive_ability

LOGGER = logging.getLogger(__name__)


def hansen_model_confidence_set(
    metrics: Mapping[str, Sequence[float]],
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    random_state: np.random.Generator | int | None = None,
    higher_is_better: bool = True,
) -> list[dict[str, float | bool]]:
    """Compute the Hansen model confidence set using SPA p-values."""
    if not metrics:
        return []

    spa_table = superior_predictive_ability(
        metrics,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        higher_is_better=higher_is_better,
    )
    if spa_table.empty:
        return []

    results: list[dict[str, float | bool]] = []
    for _, row in spa_table.iterrows():
        results.append(
            {
                "model": row["model"],
                "mean_score": float(row["mean_score"]),
                "spa_p_value": float(row["spa_p_value"]),
                "included": bool(row["spa_p_value"] > alpha),
                "ci_lower": float(row["ci_lower"]),
                "ci_upper": float(row["ci_upper"]),
            }
        )
    return results


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model confidence set analysis from a DM cache")
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
    return parser.parse_args(argv)


def _write_mcs_outputs(mcs_table: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mcs_table.to_csv(output_dir / "model_confidence_set.csv", index=False)


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

    mcs_table = tables.get("model_confidence_set", pd.DataFrame())
    _write_mcs_outputs(mcs_table, args.output_dir)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
