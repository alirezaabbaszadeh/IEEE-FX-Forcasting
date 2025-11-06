# FX Forecasting Research Plan

## Overview
This document captures prioritized research directions extracted from recent brainstorming
sessions. The goal is to guide development toward a DMLR benchmark submission while keeping
room for one or two TMLR-grade contributions.

## Immediate Priorities
1. **Purged Conformal Calibration (PCC)**
   - Implement leakage-aware conformal prediction for interval calibration.
   - Evaluate coverage at 90/95% alongside CRPS on purged validation splits.
   - Target configuration files under `configs/inference/pcc.yaml` and supporting code in
     `src/inference/conformal_purged.py` and `calibration_cli.py`.
2. **Regime-Conditioned Quantile Forecasting (RCQF)**
   - Add volatility regime labels via `src/features/regime_labels.py`.
   - Extend the lightweight LSTM architecture with a gating head in
     `src/models/deep/rcqf.py` using quantile targets τ∈{0.05, 0.5, 0.95}.
   - Track impact on CRPS, pinball loss, and coverage diagnostics.
3. **Purged Stacking Ensemble**
   - Combine ARIMA, ETS, LSTM-light, and TCN via a purged meta-learner placed in
     `src/inference/stacking_purged.py`.
   - Ensure embargoed cross-validation when generating out-of-fold predictions.

## Supporting Infrastructure
- **Calibration Diagnostics Pack**: centralize PIT histograms, coverage vs. nominal plots, and
  sharpness reports under `src/metrics/calibration.py` and `src/reporting/plots.py`.
- **Statistical Validity (SPA/MCS)**: add HAC-aware Superior Predictive Ability and Model
  Confidence Set routines within `src/stats/` to accompany benchmark tables.
- **Compute Governance**: record training and inference budgets in `paper_outputs/compute.csv`
  using utilities in `src/utils/manifest.py` to guarantee fair model comparisons.

## Evaluation Funnel
1. **Smoke Bench**: 1 pair × 1 horizon × 5 seeds, ≤3 GPU hours per model.
2. **Mini Bench**: 3 pairs × 2 horizons × 10 seeds, 2–3 days of wall-clock time.
3. **Full Bench**: 7 pairs × 3 horizons × 10 seeds for final paper artifacts.

Gate releases:
- **G1 Protocol Freeze** for DMLR submission.
- **G2 Benchmark Release** tagged `v1.0` with archival artifacts.
- **G3 Claim Freeze** preceding TMLR analyses to avoid test-set leakage.

## Success Metrics
- Achieve ≥2% average CRPS improvement or ≥2–3% coverage error reduction in at least half of
  evaluated pair-horizon cells.
- Maintain MASE while improving probabilistic calibration.
- Pass SPA/MCS tests for the promoted method without violating compute budgets.

## Risk Mitigation
- Create `tests/test_leak.py` to monitor purged split integrity.
- Enforce training time and hyperparameter sweep caps to curb compute drift.
- Apply quantile monotonicity correction via `src/inference/quantile_fix.py` to remove interval
  crossings when necessary.

