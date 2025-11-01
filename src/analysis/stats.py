"""Statistical testing utilities for experiment analysis."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    from scipy import stats as scipy_stats  # type: ignore
except ImportError:  # pragma: no cover - fallback
    scipy_stats = None

def _rankdata(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)

    # handle ties by averaging ranks
    unique_vals, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for idx, count in enumerate(counts):
        if count > 1:
            tie_positions = np.where(inverse == idx)[0]
            mean_rank = ranks[tie_positions].mean()
            ranks[tie_positions] = mean_rank
    return ranks


def one_way_anova(metrics: Mapping[str, Sequence[float]]) -> Dict[str, float]:
    groups = [np.asarray(values, dtype=float) for values in metrics.values()]
    sizes = [len(g) for g in groups]
    means = [float(g.mean()) for g in groups]
    overall_mean = float(np.mean(np.concatenate(groups)))
    ss_between = sum(size * (mean - overall_mean) ** 2 for size, mean in zip(sizes, means))
    ss_within = sum(((group - mean) ** 2).sum() for group, mean in zip(groups, means))
    df_between = len(groups) - 1
    df_within = sum(sizes) - len(groups)
    ms_between = ss_between / df_between if df_between > 0 else float("nan")
    ms_within = ss_within / df_within if df_within > 0 else float("nan")
    f_stat = ms_between / ms_within if ms_within > 0 else float("nan")

    if scipy_stats:
        p_value = float(1 - scipy_stats.f.cdf(f_stat, df_between, df_within))
    else:
        p_value = float("nan")

    ss_total = ss_between + ss_within
    eta_squared = ss_between / ss_total if ss_total else float("nan")
    omega_squared = (ss_between - df_between * ms_within) / (ss_total + ms_within) if ss_total else float("nan")

    return {
        "f_stat": float(f_stat),
        "p_value": p_value,
        "df_between": float(df_between),
        "df_within": float(df_within),
        "eta_squared": float(eta_squared),
        "omega_squared": float(omega_squared),
    }


def welch_anova(metrics: Mapping[str, Sequence[float]]) -> Dict[str, float]:
    groups = [np.asarray(values, dtype=float) for values in metrics.values()]
    means = np.array([g.mean() for g in groups])
    variances = np.array([g.var(ddof=1) for g in groups])
    sizes = np.array([len(g) for g in groups])
    weights = sizes / np.maximum(variances, 1e-8)
    weighted_mean = float(np.sum(weights * means) / np.sum(weights))
    df_num = len(groups) - 1
    numerator = np.sum(weights * (means - weighted_mean) ** 2) / df_num
    denominator = 1 + (2 * (len(groups) - 2) / (len(groups) ** 2 - 1)) * np.sum(
        ((1 - (weights / np.sum(weights))) ** 2) / (sizes - 1 + 1e-8)
    )
    f_stat = numerator / denominator

    df_denom = (len(groups) ** 2 - 1) / (3 * np.sum(((1 - weights / np.sum(weights)) ** 2) / (sizes - 1 + 1e-8)))

    if scipy_stats:
        p_value = float(1 - scipy_stats.f.cdf(f_stat, df_num, df_denom))
    else:
        p_value = float("nan")

    return {
        "f_stat": float(f_stat),
        "p_value": p_value,
        "df_num": float(df_num),
        "df_denom": float(df_denom),
    }


def kruskal_wallis(metrics: Mapping[str, Sequence[float]]) -> Dict[str, float]:
    groups = [np.asarray(values, dtype=float) for values in metrics.values()]
    concatenated = np.concatenate(groups)
    ranks = _rankdata(concatenated)
    start = 0
    rank_sums: List[float] = []
    for group in groups:
        end = start + len(group)
        rank_sums.append(float(ranks[start:end].sum()))
        start = end

    n = len(concatenated)
    h_stat = (12 / (n * (n + 1))) * sum((rank_sum**2) / len(group) for rank_sum, group in zip(rank_sums, groups)) - 3 * (n + 1)
    df = len(groups) - 1

    if scipy_stats:
        p_value = float(1 - scipy_stats.chi2.cdf(h_stat, df))
    else:
        p_value = float("nan")

    eta_squared = (h_stat - df) / (n - 1) if n > 1 else float("nan")

    return {"h_stat": float(h_stat), "p_value": p_value, "df": float(df), "eta_squared": float(eta_squared)}


def diebold_mariano(errors_a: Sequence[float], errors_b: Sequence[float], power: int = 2) -> Dict[str, float]:
    e_a = np.asarray(errors_a, dtype=float)
    e_b = np.asarray(errors_b, dtype=float)
    if e_a.shape != e_b.shape:
        raise ValueError("Error sequences must share the same length")

    if power == 2:
        losses_a = e_a ** 2
        losses_b = e_b ** 2
    else:
        losses_a = np.abs(e_a) ** power
        losses_b = np.abs(e_b) ** power

    diff = losses_a - losses_b
    mean_diff = diff.mean()
    var_diff = diff.var(ddof=1)
    n = diff.size
    denom = math.sqrt(var_diff / n + 1e-12)
    dm_stat = mean_diff / denom if denom else float("nan")
    p_value = 2 * NormalDist().cdf(-abs(dm_stat))
    return {"dm_stat": float(dm_stat), "p_value": float(p_value), "mean_diff": float(mean_diff)}


def pairwise_effect_sizes(metrics: Mapping[str, Sequence[float]], baseline: str) -> List[Dict[str, float]]:
    baseline_values = np.asarray(metrics[baseline], dtype=float)
    results: List[Dict[str, float]] = []
    for name, values in metrics.items():
        if name == baseline:
            continue
        sample = np.asarray(values, dtype=float)
        mean_diff = sample.mean() - baseline_values.mean()
        pooled_std = math.sqrt(
            ((baseline_values.size - 1) * baseline_values.var(ddof=1) + (sample.size - 1) * sample.var(ddof=1))
            / (baseline_values.size + sample.size - 2)
        )
        pooled_std = max(pooled_std, 1e-12)
        d = mean_diff / pooled_std
        correction = 1 - 3 / (4 * (baseline_values.size + sample.size) - 9)
        hedges_g = d * correction
        glass_delta = mean_diff / max(baseline_values.std(ddof=1), 1e-12)

        combined = np.concatenate([baseline_values, sample])
        ranks = _rankdata(combined)
        rank_sum_baseline = ranks[: baseline_values.size].sum()
        u_stat = rank_sum_baseline - (baseline_values.size * (baseline_values.size + 1)) / 2
        rank_biserial = 1 - (2 * u_stat) / (baseline_values.size * sample.size)

        results.append(
            {
                "comparison": f"{name} vs {baseline}",
                "cohens_d": float(d),
                "hedges_g": float(hedges_g),
                "glass_delta": float(glass_delta),
                "rank_biserial": float(rank_biserial),
            }
        )
    return results


@dataclass
class StatisticalAnalyzer:
    """Run a battery of statistical tests and persist the resulting tables."""

    run_id: str
    output_dir: Path | str = Path("artifacts")

    def analyze(self, metrics: Mapping[str, Sequence[float]], baseline: str) -> Dict[str, pd.DataFrame]:
        stats_root = Path(self.output_dir) / self.run_id / "stats"
        stats_root.mkdir(parents=True, exist_ok=True)

        anova_res = one_way_anova(metrics)
        welch_res = welch_anova(metrics)
        kruskal_res = kruskal_wallis(metrics)
        effect_res = pairwise_effect_sizes(metrics, baseline)

        dm_tables: List[Dict[str, float]] = []
        baseline_values = metrics[baseline]
        for name, values in metrics.items():
            if name == baseline:
                continue
            dm_tables.append({"comparison": f"{name} vs {baseline}", **diebold_mariano(baseline_values, values)})

        tables = {
            "anova": pd.DataFrame([anova_res]),
            "welch": pd.DataFrame([welch_res]),
            "kruskal": pd.DataFrame([kruskal_res]),
            "effect_sizes": pd.DataFrame(effect_res),
            "diebold_mariano": pd.DataFrame(dm_tables),
        }

        for name, df in tables.items():
            df.to_csv(stats_root / f"{name}.csv", index=False)

        return tables

