"""Statistical testing utilities for experiment analysis."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import logging

from statistics import NormalDist

import matplotlib

matplotlib.use("Agg", force=True)  # pragma: no cover - headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.stats.dm import construct_dm_comparisons, diebold_mariano
from src.stats.mcs import hansen_model_confidence_set
from src.stats.utils import rng_from_state

LOGGER = logging.getLogger(__name__)

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

    rng = rng_from_state(random_state)
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


def log_assumption_diagnostics(
    dm_cache: pd.DataFrame,
    *,
    metric: str = "squared_error",
    alpha: float = 0.05,
) -> None:
    if dm_cache.empty:
        LOGGER.warning("DM cache is empty; skipping assumption diagnostics")
        return
    if metric not in dm_cache.columns:
        LOGGER.warning("Metric '%s' not present in DM cache; skipping assumption diagnostics", metric)
        return
    if scipy_stats is None:
        LOGGER.warning("SciPy is unavailable; cannot evaluate normality or homoscedasticity assumptions")
        return

    grouped = dm_cache.groupby(["pair", "horizon"], dropna=False)
    for (pair, horizon), slice_df in grouped:
        LOGGER.info("Assumption checks for pair=%s horizon=%s (metric=%s)", pair, horizon, metric)
        model_groups: List[np.ndarray] = []
        for model, model_df in slice_df.groupby("model"):
            values = model_df[metric].dropna().to_numpy(dtype=float)
            if values.size < 3 or values.size > 5000:
                LOGGER.info(
                    "  Model %s has insufficient samples for Shapiro-Wilk (n=%d)",
                    model,
                    values.size,
                )
            else:
                shapiro = scipy_stats.shapiro(values)
                decision = (
                    "reject normality"
                    if shapiro.pvalue < alpha
                    else "fail to reject normality"
                )
                LOGGER.info(
                    "  Shapiro-Wilk (%s): W=%.4f, p=%.4g -> %s at alpha=%.2f",
                    model,
                    shapiro.statistic,
                    shapiro.pvalue,
                    decision,
                    alpha,
                )
            if values.size >= 2:
                model_groups.append(values)

        if len(model_groups) < 2:
            LOGGER.info("  Skipping Levene test (need at least two groups with >=2 samples)")
            continue

        levene = scipy_stats.levene(*model_groups)
        decision = (
            "reject equal variances"
            if levene.pvalue < alpha
            else "fail to reject equal variances"
        )
        LOGGER.info(
            "  Levene test: W=%.4f, p=%.4g -> %s at alpha=%.2f",
            levene.statistic,
            levene.pvalue,
            decision,
            alpha,
        )
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


def analyze_dm_cache(
    dm_cache: pd.DataFrame,
    *,
    run_id: str,
    output_dir: Path | str = Path("artifacts"),
    baseline_model: str,
    metric: str = "squared_error",
    alpha: float = 0.05,
    assumption_alpha: float = 0.05,
    newey_west_lag: int = 1,
    higher_is_better: bool = False,
    random_state: int | np.random.Generator | None = None,
    n_bootstrap: int = 2000,
) -> Dict[str, pd.DataFrame]:
    if dm_cache.empty:
        LOGGER.warning("DM cache is empty; skipping statistical post-processing")
        return {}
    if metric not in dm_cache.columns:
        raise KeyError(f"Metric column '{metric}' not present in DM cache")

    metric_frame = (
        dm_cache[["pair", "horizon", "model", metric]]
        .rename(columns={metric: "value"})
        .dropna(subset=["value"])
    )
    if metric_frame.empty:
        LOGGER.warning("No valid observations for metric '%s'; skipping statistical post-processing", metric)
        return {}

    log_assumption_diagnostics(dm_cache, metric=metric, alpha=assumption_alpha)

    analyzer = StatisticalAnalyzer(
        run_id=run_id,
        output_dir=output_dir,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        newey_west_lag=newey_west_lag,
        random_state=random_state,
        higher_is_better=higher_is_better,
    )
    return analyzer.analyze(metric_frame, baseline=baseline_model)

