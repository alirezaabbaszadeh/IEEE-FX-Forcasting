"""Helpers for running Purged Conformal Calibration toggles across benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Sequence

from .runner import MultiRunExperiment, VariantRunCollection, _ensure_serialisable


def run_pcc_toggle(
    run_fn: Callable[[MutableMapping[str, Any]], Mapping[str, Any]],
    *,
    run_id: str,
    pairs: Sequence[str],
    horizons: Sequence[object],
    seeds: Sequence[int],
    output_dir: Path | str = Path("artifacts"),
    base_config: Mapping[str, Any] | None = None,
    variant_field: str = "use_pcc",
    baseline: str = "pcc_off",
    variants: Mapping[str, Mapping[str, Any]] | None = None,
    improvement_thresholds: Mapping[str, object] | None = None,
    num_runs: int | None = None,
    base_seed: int | None = None,
) -> Dict[tuple[str, str], VariantRunCollection]:
    """Run PCC on/off ablations while reusing identical seeds and budgets.

    The helper iterates across ``pairs`` and ``horizons`` executing two variants by
    default: ``pcc_off`` (baseline) and ``pcc_on``. Each variant receives the same
    ``seeds`` list so that improvements in CRPS or coverage error stem purely from the
    calibration toggle. The function produces a manifest describing every
    pair/horizon cell and returns the :class:`VariantRunCollection` objects for
    downstream inspection.
    """

    if not pairs or not horizons:
        raise ValueError("At least one pair and horizon must be provided")

    seeds_list = [int(seed) for seed in seeds]
    if not seeds_list:
        raise ValueError("Seed list cannot be empty for PCC toggle experiments")

    if num_runs is None:
        num_runs = len(seeds_list)
    if num_runs <= 0:
        raise ValueError("num_runs must be a positive integer")
    if len(seeds_list) < num_runs:
        raise ValueError("Provided seeds do not cover the requested number of runs")

    if base_seed is None:
        base_seed = min(seeds_list)

    experiment = MultiRunExperiment(run_fn, num_runs=num_runs, base_seed=base_seed)

    if variants is None:
        variants = {
            "pcc_off": {variant_field: False},
            "pcc_on": {variant_field: True},
        }

    default_thresholds = {
        "crps": {"direction": "decrease", "relative": 0.02},
        "coverage_error": {"direction": "decrease", "relative": 0.02},
    }
    thresholds = improvement_thresholds or default_thresholds

    base_config = dict(base_config or {})
    output_dir = Path(output_dir)

    results: Dict[tuple[str, str], VariantRunCollection] = {}
    manifest_entries: list[dict[str, Any]] = []

    for pair in pairs:
        for horizon in horizons:
            variant_config: Dict[str, Any] = dict(base_config)
            variant_config["pair"] = pair
            variant_config["horizon"] = horizon

            collection = experiment.run_variants(
                variant_config,
                run_id=f"{run_id}/{pair}_{horizon}",
                output_dir=output_dir,
                seeds=seeds_list,
                variants=variants,
                baseline=baseline,
                improvement_thresholds=thresholds,
            )

            key = (str(pair), str(horizon))
            results[key] = collection

            summary_path = collection.summary_path
            try:
                summary_entry = str(summary_path.relative_to(output_dir))
            except ValueError:
                summary_entry = str(summary_path)

            manifest_entries.append(
                {
                    "pair": pair,
                    "horizon": str(horizon),
                    "baseline": collection.baseline,
                    "summary": summary_entry,
                }
            )

    manifest = {
        "run_id": run_id,
        "pairs": list(pairs),
        "horizons": [str(h) for h in horizons],
        "seeds": seeds_list,
        "variants": _ensure_serialisable(variants),
        "baseline": baseline,
        "thresholds": _ensure_serialisable(thresholds),
        "entries": manifest_entries,
    }

    manifest_path = output_dir / run_id / "pcc_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_ensure_serialisable(manifest), indent=2))

    return results


__all__ = ["run_pcc_toggle"]

