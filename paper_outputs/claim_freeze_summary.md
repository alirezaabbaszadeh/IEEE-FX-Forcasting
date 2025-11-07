# Claim Freeze Validation Summary

## PCC improvements against baselines
- `scripts/reproduce_all.sh` now evaluates every prediction file with purged conformal calibration
  before assembling paper assets. This run step generates calibrated intervals, DM caches, and
  statistical manifests inside the originating run directories while honouring the freeze manifest.
- The statistical battery in `src/analysis/stats.py` emits HAC-aware DM, SPA, and MCS tables for each
  pair×horizon slice and for every discovered volatility regime or event segment. Segment manifests are
  deposited alongside the standard `paper_outputs/stats` tables so reviewers can audit performance in
  stress regimes without recomputing the sweep.

## Consistency with the frozen claim
- `src/eval/run.py` refuses to load test rows until the claim freeze manifest has been acknowledged and
  confirms that the earliest test timestamp is at or after the recorded freeze. The manifest itself is
  stored at `configs/governance/claim_freeze.yaml` and is copied into each evaluation output folder for
  provenance.
- Calibration diagnostics produced by `src/metrics/calibration_cli.py` now emit PIT histograms and
  coverage plots for the overall distribution, volatility regimes, and any labelled events. The plots
  are written to `paper_outputs/calibration/figs`, allowing easy comparison between baseline and PCC
  variants while maintaining the freeze.
