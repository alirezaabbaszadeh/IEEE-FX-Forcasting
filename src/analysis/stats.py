"""Statistical testing utilities for experiment analysis."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)  # pragma: no cover - headless execution
import matplotlib.pyplot as plt
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


def _rng_from_state(random_state: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        seed = int(random_state.integers(0, np.iinfo(np.int64).max))
        return np.random.default_rng(seed)
    if random_state is None:
        return np.random.default_rng()
    return np.random.default_rng(random_state)


def bootstrap_confidence_interval(
    values: Sequence[float],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    statistic: callable = np.mean,
    random_state: np.random.Generator | int | None = None,
) -> Dict[str, float]:
    sample = np.asarray(values, dtype=float)
    if sample.size == 0:
        return {"estimate": float("nan"), "lower": float("nan"), "upper": float("nan")}

    if n_resamples <= 0:
        estimate = float(statistic(sample))
        return {"estimate": estimate, "lower": estimate, "upper": estimate}

    rng = _rng_from_state(random_state)
    estimates = np.empty(n_resamples, dtype=float)
    for idx in range(n_resamples):
        resample = rng.choice(sample, size=sample.size, replace=True)
        estimates[idx] = float(statistic(resample))

    estimate = float(statistic(sample))
    lower = float(np.quantile(estimates, alpha / 2))
    upper = float(np.quantile(estimates, 1 - alpha / 2))
    return {"estimate": estimate, "lower": lower, "upper": upper}


def _studentized_range_sf(q_stat: float, k: int, df: float) -> float:
    if scipy_stats and hasattr(scipy_stats, "studentized_range"):
        return float(scipy_stats.studentized_range.sf(q_stat, k, df))
    if scipy_stats and hasattr(scipy_stats, "tukey_hsd"):
        # use scipy's tukey to approximate survival function by inversion
        res = scipy_stats.tukey_hsd([0.0] * k, [1] * k, df)
        if hasattr(res, "pvalues"):
            return float(res.pvalues[0])
    if scipy_stats:
        return float(2 * (1 - scipy_stats.t.cdf(q_stat / math.sqrt(2), df)))
    return float("nan")


def tukey_hsd(
    metrics: Mapping[str, Sequence[float]],
    *,
    alpha: float = 0.05,
) -> List[Dict[str, float]]:
    arrays = {name: np.asarray(values, dtype=float) for name, values in metrics.items()}
    n_groups = len(arrays)
    if n_groups < 2:
        return []

    mse_num = 0.0
    total_n = 0
    for sample in arrays.values():
        total_n += sample.size
        mean = sample.mean() if sample.size else 0.0
        mse_num += float(((sample - mean) ** 2).sum())
    df_within = total_n - n_groups
    mse = mse_num / df_within if df_within > 0 else float("nan")

    q_crit = float("nan")
    if scipy_stats and hasattr(scipy_stats, "studentized_range") and df_within > 0:
        q_crit = float(scipy_stats.studentized_range.ppf(1 - alpha, n_groups, df_within))

    results: List[Dict[str, float]] = []
    for name_a, name_b in combinations(arrays.keys(), 2):
        sample_a = arrays[name_a]
        sample_b = arrays[name_b]
        n_a = sample_a.size
        n_b = sample_b.size
        mean_diff = float(sample_a.mean() - sample_b.mean())
        se = math.sqrt(mse / 2 * (1 / n_a + 1 / n_b)) if n_a and n_b else float("nan")
        q_stat = abs(mean_diff) / se if se else float("nan")
        p_value = _studentized_range_sf(q_stat, n_groups, df_within) if not math.isnan(q_stat) else float("nan")
        half_width = q_crit * se if not math.isnan(q_crit) and se else float("nan")
        results.append(
            {
                "model_a": name_a,
                "model_b": name_b,
                "mean_diff": mean_diff,
                "q_stat": float(q_stat),
                "p_value": float(p_value),
                "reject": bool(p_value < alpha) if not math.isnan(p_value) else False,
                "ci_lower": float(mean_diff - half_width) if not math.isnan(half_width) else float("nan"),
                "ci_upper": float(mean_diff + half_width) if not math.isnan(half_width) else float("nan"),
            }
        )
    return results


def dunn_posthoc(
    metrics: Mapping[str, Sequence[float]],
    *,
    alpha: float = 0.05,
    correction: str = "holm",
) -> List[Dict[str, float]]:
    arrays = {name: np.asarray(values, dtype=float) for name, values in metrics.items()}
    groups = list(arrays.items())
    if len(groups) < 2:
        return []

    concatenated = np.concatenate([values for _, values in groups])
    ranks = _rankdata(concatenated)
    n = concatenated.size
    start = 0
    mean_ranks: Dict[str, float] = {}
    for name, values in groups:
        end = start + values.size
        mean_ranks[name] = float(ranks[start:end].mean())
        start = end

    tie_correction = 0.0
    _, counts = np.unique(concatenated, return_counts=True)
    tie_correction = 1.0 - (counts**3 - counts).sum() / (n**3 - n) if n > 1 else 1.0
    var_factor = n * (n + 1) / 12.0

    raw_p: List[float] = []
    rows: List[Dict[str, float]] = []
    for (name_a, values_a), (name_b, values_b) in combinations(groups, 2):
        diff = mean_ranks[name_a] - mean_ranks[name_b]
        se = math.sqrt(var_factor * (1 / values_a.size + 1 / values_b.size) * tie_correction)
        z_stat = diff / se if se else float("nan")
        p_value = 2 * NormalDist().cdf(-abs(z_stat)) if not math.isnan(z_stat) else float("nan")
        raw_p.append(p_value)
        rows.append(
            {
                "model_a": name_a,
                "model_b": name_b,
                "z_stat": float(z_stat),
                "p_value": float(p_value),
            }
        )

    if correction.lower() == "holm":
        order = sorted(range(len(raw_p)), key=lambda idx: raw_p[idx])
        m = len(raw_p)
        running_max = 0.0
        adjusted = [0.0] * m
        for rank, idx in enumerate(order):
            factor = m - rank
            value = raw_p[idx] * factor
            running_max = max(running_max, value)
            adjusted[idx] = min(1.0, running_max)
        for idx, row in enumerate(rows):
            row["p_adjusted"] = float(adjusted[idx])
            row["reject"] = bool(adjusted[idx] < alpha)
    else:
        for idx, row in enumerate(rows):
            row["p_adjusted"] = float(raw_p[idx])
            row["reject"] = bool(raw_p[idx] < alpha)
    return rows


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


def hansen_model_confidence_set(
    metrics: Mapping[str, Sequence[float]],
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    random_state: np.random.Generator | int | None = None,
    higher_is_better: bool = True,
) -> List[Dict[str, float]]:
    arrays = {name: np.asarray(values, dtype=float) for name, values in metrics.items()}
    if not arrays:
        return []

    lengths = [sample.size for sample in arrays.values() if sample.size]
    if not lengths:
        return []
    min_len = min(lengths)
    trimmed = {name: sample[:min_len] for name, sample in arrays.items()}
    matrix = np.column_stack([values for values in trimmed.values()])
    if not higher_is_better:
        losses = matrix
    else:
        losses = -matrix

    means = losses.mean(axis=0)
    best_idx = int(np.argmin(means))
    diff_obs = means - means[best_idx]

    if n_bootstrap <= 0 or min_len < 2:
        return [
            {
                "model": name,
                "mean_score": float(trimmed[name].mean()),
                "spa_p_value": 1.0 if idx == best_idx else 0.0,
                "included": bool(idx == best_idx),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
            }
            for idx, name in enumerate(trimmed.keys())
        ]

    rng = _rng_from_state(random_state)
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

    results: List[Dict[str, float]] = []
    for idx, name in enumerate(trimmed.keys()):
        results.append(
            {
                "model": name,
                "mean_score": float(trimmed[name].mean()),
                "spa_p_value": float(p_values[idx]),
                "included": bool(p_values[idx] > alpha),
                "ci_lower": float(ci_lower[idx]),
                "ci_upper": float(ci_upper[idx]),
            }
        )
    return results


def _sanitize_slug(*parts: str) -> str:
    slug = "-".join(str(part) for part in parts if part is not None)
    slug = re.sub(r"[^a-zA-Z0-9\-_.]+", "_", slug)
    return slug or "slice"


def collect_multirun_repository(
    root: Path | str,
    metric: str,
    *,
    pair_name: str = "pair",
    horizon_name: str = "horizon",
) -> pd.DataFrame:
    base = Path(root)
    rows: List[Dict[str, object]] = []
    for model_dir in base.rglob("*"):
        if not model_dir.is_dir():
            continue
        seed_dirs = list(model_dir.glob("seed-*/metrics.json"))
        if not seed_dirs:
            continue
        parts = model_dir.relative_to(base).parts
        if len(parts) == 3:
            pair, horizon, model = parts
        elif len(parts) == 2:
            pair, model = parts
            horizon = "global"
        else:
            continue
        for metrics_path in seed_dirs:
            try:
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if metric not in payload:
                continue
            seed = metrics_path.parent.name.replace("seed-", "")
            rows.append(
                {
                    pair_name: pair,
                    horizon_name: horizon,
                    "model": model,
                    "seed": seed,
                    "value": float(payload[metric]),
                }
            )
    return pd.DataFrame(rows)


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
    rows: List[Dict[str, object]] = []
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
    n_bootstrap: int = 2000
    alpha: float = 0.05
    newey_west_lag: int = 1
    random_state: int | np.random.Generator | None = None
    higher_is_better: bool = True

    def _normalize_metrics(
        self,
        metrics: Mapping[str, Sequence[float]]
        | Mapping[Tuple[str, str], Mapping[str, Sequence[float]]]
        | pd.DataFrame,
    ) -> List[Tuple[str, str, Dict[str, Sequence[float]]]]:
        if isinstance(metrics, pd.DataFrame):
            frame = metrics.copy()
            if "pair" not in frame.columns:
                frame["pair"] = "global"
            if "horizon" not in frame.columns:
                frame["horizon"] = "global"
            required = {"model", "value"}
            if not required.issubset(frame.columns):
                raise ValueError("DataFrame metrics must include model and value columns")
            result: List[Tuple[str, str, Dict[str, Sequence[float]]]] = []
            for (pair, horizon), group in frame.groupby(["pair", "horizon"], dropna=False):
                mapping: Dict[str, List[float]] = {}
                for model, value_series in group.groupby("model"):
                    mapping[str(model)] = [float(v) for v in value_series["value"].tolist()]
                result.append((str(pair), str(horizon), mapping))
            return result

        normalized: List[Tuple[str, str, Dict[str, Sequence[float]]]] = []
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping or DataFrame")

        for key, value in metrics.items():
            if isinstance(value, Mapping):
                if isinstance(key, tuple) and len(key) == 2:
                    pair, horizon = key
                else:
                    pair, horizon = str(key), "global"
                normalized.append((str(pair), str(horizon), {k: list(v) for k, v in value.items()}))
            else:
                normalized.append(("global", "global", {k: list(v) for k, v in metrics.items()}))
                break
        return normalized

    def _write_tables(self, stats_root: Path, tables: Mapping[str, pd.DataFrame]) -> None:
        for name, df in tables.items():
            df.to_csv(stats_root / f"{name}.csv", index=False)
            df.to_json(stats_root / f"{name}.json", orient="records", indent=2)

    def _plot_confidence_bands(self, stats_root: Path, bootstrap_rows: Mapping[Tuple[str, str], pd.DataFrame]) -> None:
        for (pair, horizon), df in bootstrap_rows.items():
            if df.empty:
                continue
            slug = _sanitize_slug(pair, horizon)
            fig, ax = plt.subplots(figsize=(6, 4))
            df_sorted = df.sort_values("model")
            x = np.arange(len(df_sorted))
            estimates = df_sorted["estimate"].to_numpy()
            lower = df_sorted["lower"].to_numpy()
            upper = df_sorted["upper"].to_numpy()
            ax.plot(x, estimates, marker="o", label="Estimate")
            ax.fill_between(x, lower, upper, color="tab:blue", alpha=0.2, label="CI band")
            ax.set_xticks(x)
            ax.set_xticklabels(df_sorted["model"], rotation=45, ha="right")
            ax.set_title(f"Bootstrap CI ({pair}, {horizon})")
            ax.set_ylabel("Metric")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            for ext in ("pdf", "svg"):
                fig.savefig(stats_root / f"ci_bands_{slug}.{ext}")
            plt.close(fig)

    def analyze(
        self,
        metrics: Mapping[str, Sequence[float]]
        | Mapping[Tuple[str, str], Mapping[str, Sequence[float]]]
        | pd.DataFrame,
        baseline: str,
    ) -> Dict[str, pd.DataFrame]:
        stats_root = Path(self.output_dir) / self.run_id / "stats"
        stats_root.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(self.random_state)
        slices = self._normalize_metrics(metrics)

        anova_rows: List[Dict[str, object]] = []
        welch_rows: List[Dict[str, object]] = []
        kruskal_rows: List[Dict[str, object]] = []
        effect_rows: List[Dict[str, object]] = []
        dm_rows: List[Dict[str, object]] = []
        bootstrap_rows: Dict[Tuple[str, str], pd.DataFrame] = {}
        tukey_rows: List[Dict[str, object]] = []
        dunn_rows: List[Dict[str, object]] = []
        mcs_rows: List[Dict[str, object]] = []

        for pair, horizon, mapping in slices:
            if baseline not in mapping:
                raise KeyError(f"Baseline '{baseline}' not present in metrics for slice {(pair, horizon)}")

            anova = one_way_anova(mapping)
            welch = welch_anova(mapping)
            kruskal = kruskal_wallis(mapping)
            anova_rows.append({"pair": pair, "horizon": horizon, **anova})
            welch_rows.append({"pair": pair, "horizon": horizon, **welch})
            kruskal_rows.append({"pair": pair, "horizon": horizon, **kruskal})

            effects = pairwise_effect_sizes(mapping, baseline)
            for row in effects:
                effect_rows.append({"pair": pair, "horizon": horizon, **row})

            boot_records: List[Dict[str, object]] = []
            for name, values in mapping.items():
                ci = bootstrap_confidence_interval(
                    values,
                    n_resamples=self.n_bootstrap,
                    alpha=self.alpha,
                    random_state=rng,
                )
                coverage = float(np.mean((np.asarray(values) >= ci["lower"]) & (np.asarray(values) <= ci["upper"])))
                boot_records.append(
                    {
                        "pair": pair,
                        "horizon": horizon,
                        "model": name,
                        "estimate": ci["estimate"],
                        "lower": ci["lower"],
                        "upper": ci["upper"],
                        "coverage": coverage,
                    }
                )
            bootstrap_rows[(pair, horizon)] = pd.DataFrame(boot_records)

            for name_a, name_b in combinations(mapping.keys(), 2):
                dm = diebold_mariano(
                    mapping[name_a],
                    mapping[name_b],
                    power=2,
                    use_newey_west=True,
                    lag=self.newey_west_lag,
                )
                dm_rows.append(
                    {
                        "pair": pair,
                        "horizon": horizon,
                        "model_a": name_a,
                        "model_b": name_b,
                        **dm,
                    }
                )

            tukey_res = tukey_hsd(mapping, alpha=self.alpha)
            for row in tukey_res:
                tukey_rows.append({"pair": pair, "horizon": horizon, **row})

            dunn_res = dunn_posthoc(mapping, alpha=self.alpha)
            for row in dunn_res:
                dunn_rows.append({"pair": pair, "horizon": horizon, **row})

            mcs_res = hansen_model_confidence_set(
                mapping,
                alpha=self.alpha,
                n_bootstrap=self.n_bootstrap,
                random_state=rng,
                higher_is_better=self.higher_is_better,
            )
            for row in mcs_res:
                mcs_rows.append({"pair": pair, "horizon": horizon, **row})

        tables = {
            "anova": pd.DataFrame(anova_rows),
            "welch": pd.DataFrame(welch_rows),
            "kruskal": pd.DataFrame(kruskal_rows),
            "effect_sizes": pd.DataFrame(effect_rows),
            "diebold_mariano": pd.DataFrame(dm_rows),
            "bootstrap_ci": pd.concat(bootstrap_rows.values(), ignore_index=True) if bootstrap_rows else pd.DataFrame(),
            "posthoc_tukey": pd.DataFrame(tukey_rows),
            "posthoc_dunn": pd.DataFrame(dunn_rows),
            "model_confidence_set": pd.DataFrame(mcs_rows),
        }

        self._write_tables(stats_root, tables)
        self._plot_confidence_bands(stats_root, bootstrap_rows)

        return tables

    def analyze_repository(
        self,
        repository_root: Path | str,
        metric: str,
        baseline: str,
    ) -> Dict[str, pd.DataFrame]:
        repo = collect_multirun_repository(repository_root, metric)
        tables = self.analyze(repo, baseline)
        dm_table = construct_dm_comparisons(repo, lag=self.newey_west_lag)
        stats_root = Path(self.output_dir) / self.run_id / "stats"
        dm_table.to_csv(stats_root / "diebold_mariano_repository.csv", index=False)
        dm_table.to_json(stats_root / "diebold_mariano_repository.json", orient="records", indent=2)
        tables["diebold_mariano_repository"] = dm_table
        return tables

