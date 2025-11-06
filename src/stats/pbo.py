"""Probability of backtest overfitting utilities and CLI entry point."""
from __future__ import annotations

import argparse
import logging
import math
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.backtest.engine import append_metrics_to_pbo_table

from .utils import rng_from_state

LOGGER = logging.getLogger(__name__)


def _sample_combinations(
    n_obs: int,
    train_size: int,
    max_combinations: int,
    rng: np.random.Generator,
) -> Iterable[tuple[int, ...]]:
    total = math.comb(n_obs, train_size)
    if total <= max_combinations:
        yield from combinations(range(n_obs), train_size)
        return

    seen: set[tuple[int, ...]] = set()
    while len(seen) < max_combinations:
        choice = tuple(sorted(rng.choice(n_obs, size=train_size, replace=False).tolist()))
        if choice in seen:
            continue
        seen.add(choice)
        yield choice


def probability_of_backtest_overfitting(
    dm_cache: pd.DataFrame,
    *,
    metric: str,
    higher_is_better: bool = False,
    max_combinations: int = 128,
    random_state: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """Estimate the probability of backtest overfitting using combinatorial CV."""
    if metric not in dm_cache.columns:
        raise KeyError(f"Metric column '{metric}' not present in DM cache")

    rng = rng_from_state(random_state)
    rows: list[dict[str, object]] = []

    grouped = dm_cache.groupby(["pair", "horizon"], dropna=False)
    for (pair, horizon), group in grouped:
        pivot = (
            group.pivot_table(index="timestamp", columns="model", values=metric)
            .dropna(axis=0, how="any")
        )
        n_obs = pivot.shape[0]
        if n_obs < 2:
            continue
        train_size = n_obs // 2
        test_size = n_obs - train_size
        if train_size == 0 or test_size == 0:
            continue

        overfit_count = 0
        total_trials = 0
        for combo in _sample_combinations(n_obs, train_size, max_combinations, rng):
            mask = np.zeros(n_obs, dtype=bool)
            mask[list(combo)] = True
            train_values = pivot.values[mask]
            test_values = pivot.values[~mask]
            if train_values.size == 0 or test_values.size == 0:
                continue

            train_scores = train_values.mean(axis=0)
            test_scores = test_values.mean(axis=0)
            if higher_is_better:
                best_idx = int(np.argmax(train_scores))
                threshold = float(np.median(test_scores))
                overfit = bool(test_scores[best_idx] < threshold)
            else:
                best_idx = int(np.argmin(train_scores))
                threshold = float(np.median(test_scores))
                overfit = bool(test_scores[best_idx] > threshold)
            overfit_count += int(overfit)
            total_trials += 1

        if total_trials == 0:
            continue

        pbo = overfit_count / total_trials
        rows.append(
            {
                "pair": pair,
                "horizon": horizon,
                "pbo": float(pbo),
                "overfit_trials": int(overfit_count),
                "total_trials": int(total_trials),
            }
        )

    return pd.DataFrame(rows)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate probability of backtest overfitting from a DM cache")
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
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=128,
        help="Maximum number of combinatorial splits sampled per slice",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000, help="Number of bootstrap samples")
    return parser.parse_args(argv)


def _write_pbo_outputs(pbo_table: pd.DataFrame, output_dir: Path, *, run_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pbo_table.to_csv(output_dir / "pbo.csv", index=False)
    if not pbo_table.empty:
        metrics = pbo_table.assign(
            run_id=run_id,
            scenario="pbo",
            metric="pbo",
            value=pbo_table["pbo"],
        )[["run_id", "pair", "horizon", "scenario", "metric", "value"]]
        append_metrics_to_pbo_table(metrics)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    dm_cache = pd.read_csv(args.dm_cache)

    from src.analysis.stats import analyze_dm_cache

    analyze_dm_cache(
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

    pbo_table = probability_of_backtest_overfitting(
        dm_cache,
        metric=args.metric,
        higher_is_better=args.higher_is_better,
        max_combinations=args.max_combinations,
        random_state=args.random_seed,
    )

    _write_pbo_outputs(pbo_table, args.output_dir, run_id=str(args.run_id))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
