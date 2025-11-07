# Purged Conformal Calibration Ablations

## Embargo Settings
- PCC toggles now reuse the exact seed schedule across all pair–horizon cells via the
  `run_pcc_toggle` helper, ensuring embargo adjustments are isolated from sampling noise.
- The variant manifest (`variants.json`) captures embargo-dependent coverage deltas so that
  aggregation checks flag any configuration missing the ≥2% improvement bar prior to a claim
  freeze.
- Baseline manifests enumerate the embargo budget used for each pair/horizon combination,
  enabling quick audits when embargo widening is required.

## Weighting Schemes
- Recency weighting changes are encoded as variant overrides, allowing the shared seed and
  compute budget to remain constant when calibrator decay factors are swept.
- Aggregated outputs now include CRPS and coverage deltas for each weighting option, making it
  clear when exponential decay outperforms uniform weights by the mandated ≥2% margin.
- Summary manifests under `artifacts/*/variants.json` document the exact weighting parameters
  evaluated, simplifying reproduction of the ablation cells reported here.
