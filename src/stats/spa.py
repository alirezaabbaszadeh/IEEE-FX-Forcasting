"""Superior Predictive Ability utilities and CLI entry point."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .utils import rng_from_state

LOGGER = logging.getLogger(__name__)


def _prepare_loss_matrix(
    metrics: Mapping[str, Sequence[float]],
    *,
    higher_is_better: bool,
) -> tuple[list[str], np.ndarray, int]:
    arrays = {name: np.asarray(values, dtype=float) for name, values in metrics.items()}
    if not arrays:
        return [], np.empty((0, 0)), 0

    lengths = [sample.size for sample in arrays.values() if sample.size]
    if not lengths:
        return [], np.empty((0, 0)), 0

    min_len = min(lengths)
    trimmed = {name: sample[:min_len] for name, sample in arrays.items()}
    matrix = np.column_stack([values for values in trimmed.values()])
    if higher_is_better:
        losses = -matrix
    else:
        losses = matrix
    names = list(trimmed.keys())
    return names, losses, min_len


def superior_predictive_ability(
    metrics: Mapping[str, Sequence[float]],
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    random_state: np.random.Generator | int | None = None,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Compute SPA p-values for a mapping of model metrics."""
    names, losses, min_len = _prepare_loss_matrix(metrics, higher_is_better=higher_is_better)
    if not names or min_len == 0:
        return pd.DataFrame(columns=["model", "mean_score", "spa_p_value", "ci_lower", "ci_upper"])

    means = losses.mean(axis=0)
    best_idx = int(np.argmin(means))
    diff_obs = means - means[best_idx]

    trimmed_scores = {name: np.asarray(metrics[name], dtype=float)[:min_len] for name in names}

    if n_bootstrap <= 0 or min_len < 2:
        rows = []
        for idx, name in enumerate(names):
            values = trimmed_scores[name]
            rows.append(
                {
                    "model": name,
                    "mean_score": float(values.mean()),
                    "spa_p_value": 1.0 if idx == best_idx else 0.0,
                    "ci_lower": float("nan"),
                    "ci_upper": float("nan"),
                }
            )
        return pd.DataFrame(rows)

    rng = rng_from_state(random_state)
    boot_diffs = np.empty((n_bootstrap, diff_obs.size), dtype=float)
    for b in range(n_bootstrap):
        indices = rng.integers(0, min_len, size=min_len)
        sampled = losses[indices]
        sample_means = sampled.mean(axis=0)
        boot_best = sample_means.min()
        boot_diffs[b] = sample_means - boot_best

    p_values = np.mean(boot_diffs >= diff_obs, axis=0)
    ci_upper = np.quantile(boot_diffs, 1 - alpha, axis=0)
    ci_lower = np.quantile(boot_diffs, alpha, axis=0)

    rows = []
    for idx, name in enumerate(names):
        values = trimmed_scores[name]
        rows.append(
            {
                "model": name,
                "mean_score": float(values.mean()),
                "spa_p_value": float(p_values[idx]),
                "ci_lower": float(ci_lower[idx]),
                "ci_upper": float(ci_upper[idx]),
            }
        )
    return pd.DataFrame(rows)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SPA analysis from a DM cache")
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


def _write_spa_outputs(spa_table: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    spa_table.to_csv(output_dir / "spa.csv", index=False)


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
    if mcs_table.empty:
        LOGGER.warning("Model confidence set table is empty; skipping SPA export")
        spa_table = pd.DataFrame()
    else:
        spa_table = mcs_table[["pair", "horizon", "model", "spa_p_value", "ci_lower", "ci_upper"]].copy()

    _write_spa_outputs(spa_table, args.output_dir)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
