# FX Forecasting Research Plan

## Overview
This document distills the brainstorming ideas into an actionable roadmap for extending the
IEEE FX Forecasting project toward competitive DMLR benchmarks and a potential TMLR
submission. The plan balances quick wins, medium-effort upgrades, and ambitious
experiments while enforcing rigorous evaluation and compute accounting.

## Prioritized initiatives
1. **Purged Conformal Calibration (PCC)**
   - Build conformal prediction intervals that respect purged and embargoed splits.
   - Target: ±2–3% coverage error at 90/95% while lowering CRPS compared with baseline
     quantile heads.
   - Hooks: `src/inference/conformal_purged.py`, `configs/inference/pcc.yaml`,
     `calibration_cli.py`.
2. **Regime-Conditioned Quantile Forecasting (RCQF)**
   - Add volatility regime labels and a gating head to shrink quantile error under regime
     shifts.
   - Target: 2% CRPS reduction with better high-volatility coverage and PIT uniformity.
   - Hooks: `src/features/regime_labels.py`, `src/models/deep/rcqf.py`,
     `configs/agent/rcqf.yaml`.
3. **Purged Stacking Ensemble**
   - Train a meta-learner on embargo-respecting out-of-fold predictions from diverse base
     models.
   - Target: 1–3% improvements in MASE and CRPS with SPA evidence of superiority.
   - Hooks: `src/inference/stacking_purged.py`, `configs/inference/stacking.yaml`.
4. **Calibration Diagnostics Pack**
   - Provide PIT, coverage, and sharpness plots plus CRPS/Pinball metrics for every
     experiment.
   - Hooks: `src/metrics/calibration.py`, `src/reporting/plots.py`.
5. **Compute Governance and Statistical Safeguards**
   - Enforce equal compute budgets, log resource usage, and run SPA/MCS for multi-model
     comparisons.
   - Hooks: `src/utils/manifest.py`, `paper_outputs/compute.csv`, `src/stats/spa.py`,
     `src/stats/mcs.py`.

## Experimental funnel
- **Smoke tests**: 1 pair × 1 horizon × 5 seeds (≤3 GPU hours per model).
- **Mini-benchmark**: 3 pairs × 2 horizons × 10 seeds (2–3 days total compute).
- **Full benchmark**: 7 pairs × 3 horizons × 10 seeds for publication-ready results.

## Decision gates
1. **Protocol freeze (DMLR)**: Lock splits, metrics, and compute caps.
2. **Benchmark release (v1.0)**: Archive artifacts and publish benchmark card.
3. **Claim freeze (TMLR)**: Choose a single headline method post mini-benchmark; reserve
   the final test set until this point.

## Measurement suite
- Point accuracy: MAE, RMSE, MASE.
- Probabilistic quality: CRPS, Pinball at multiple quantiles, coverage@90/95, PIT
  diagnostics.
- Statistical validation: DM with HAC, Superior Predictive Ability (SPA), Model Confidence
  Set (MCS), and optional Probability of Backtest Overfitting (PBO).
- Compute tracking: training/inference time, peak memory, improvement per compute unit.

## Risk controls
- Audit all data splits with automated tests to eliminate leakage.
- Enforce HPO and epoch caps; fail runs that exceed the budget.
- Define ablation grids up front and report SPA/MCS to avoid cherry-picking.
- Apply quantile monotonicity fixes when forecasting multiple quantiles.
- Maintain reproducible scripts (`scripts/reproduce_all.sh`) for all reported tables and
  figures.
