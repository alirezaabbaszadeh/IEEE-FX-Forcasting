# FX Forecasting Research Plan

## Overview
This document captures prioritized research directions for the IEEE FX
Forecasting benchmark. The goal is to guide development toward a DMLR benchmark
submission while leaving room for targeted TMLR-grade contributions. All work
streams below assume the consolidated `src/` platform and governance manifests
are the source of truth.

## Immediate Priorities
1. **Purged Conformal Calibration (PCC) Extensions**
   - Harden leakage-aware conformal prediction for interval calibration by
     expanding residual diagnostics and coverage reporting in
     `src/inference/conformal_purged.py` and the associated CLI.【F:src/inference/conformal_purged.py†L1-L200】
   - Evaluate coverage at 90/95% alongside CRPS on purged validation splits using
     `configs/inference/pcc.yaml` for governance defaults.【F:configs/inference/pcc.yaml†L1-L8】
   - Surface calibration manifests in `paper_outputs/ablations/pcc_ablation.md`
     whenever new weighting or embargo policies are introduced.【F:paper_outputs/ablations/pcc_ablation.md†L1-L18】
2. **Regime-Conditioned Quantile Forecasting (RCQF)**
   - Maintain volatility regime labels via `src/features/regime_labels.py` and
     integrate them with the quantile head in `src/models/deep/rcqf.py`.
     【F:src/features/regime_labels.py†L1-L160】【F:src/models/deep/rcqf.py†L1-L200】
   - Track impact on CRPS, pinball loss, and coverage diagnostics using
     `src/metrics/calibration.py` and the reporting utilities referenced in the
     benchmark card.【F:src/metrics/calibration.py†L1-L120】
3. **Purged Stacking Ensemble**
   - Combine ARIMA, ETS, LSTM-light, and TCN via the purged meta-learner in
     `src/inference/stacking_purged.py`, ensuring embargoed cross-validation when
     generating out-of-fold predictions.【F:src/inference/stacking_purged.py†L1-L200】
   - Log compute budgets and residual manifests through `src.utils.manifest` so
     ensemble claims inherit the reproducibility guarantees.【F:src/utils/manifest.py†L1-L200】

## Supporting Infrastructure
- **Calibration Diagnostics Pack:** Centralise PIT histograms, coverage vs
  nominal plots, and sharpness reports using `src/metrics/calibration.py` and
  `src/reporting/aggregates.py` so that manifests include every diagnostic used in
  publication tables.【F:src/reporting/aggregates.py†L1-L200】
- **Statistical Validity (SPA/MCS):** Maintain HAC-aware Superior Predictive
  Ability and Model Confidence Set routines within `src/analysis/stats.py` to
  accompany benchmark tables and enforce the ≥2% claim gates.【F:src/analysis/stats.py†L1-L220】
- **Compute Governance:** Record training and inference budgets in
  `paper_outputs/compute.csv` using utilities in `src/utils/manifest.py` to ensure
  fair model comparisons.【F:src/utils/manifest.py†L1-L200】

## Evaluation Funnel
1. **Smoke Bench:** 1 pair × 1 horizon × 5 seeds, ≤3 GPU hours per model.
2. **Mini Bench:** 3 pairs × 2 horizons × 10 seeds, 2–3 days of wall-clock time.
3. **Full Bench:** 7 pairs × 3 horizons × 10 seeds for final publication assets.

Gate releases:
- **G1 Protocol Freeze** for DMLR submission.
- **G2 Benchmark Release** tagged `v1.0` with archival artifacts.
- **G3 Claim Freeze** preceding TMLR analyses to avoid test-set leakage.

## Success Metrics
- Achieve ≥2% average CRPS improvement or ≥2–3% coverage error reduction in at
  least half of evaluated pair–horizon cells.
- Maintain MASE while improving probabilistic calibration.
- Pass SPA/MCS tests for the promoted method without violating compute budgets.

## Risk Mitigation
- Keep `tests/test_leak.py` and related regressions green to monitor purged split
  integrity.【F:tests/test_leak.py†L1-L74】
- Enforce training time and hyperparameter sweep caps via the governance limits
  in `configs/default.yaml` and friends.【F:configs/default.yaml†L1-L80】
- Apply quantile monotonicity correction via `src/inference/quantile_fix.py` to
  remove interval crossings when necessary.【F:src/inference/quantile_fix.py†L1-L160】
