# IEEE FX Forecasting Benchmark Card

## Overview
The IEEE FX Forecasting benchmark provides a reproducible evaluation suite for
mix-of-experts sequence models and classical baselines on short-horizon foreign
exchange forecasting tasks. It packages curated datasets, deterministic
walk-forward splits, and reporting scripts so that published metrics can be
replicated and audited by third parties.

## Evaluation protocol
- **Windowed walk-forward splits.** Each currency pair and prediction horizon is
  evaluated with sequential walk-forward windows that honour configurable
  embargo periods and never reuse timestamps across train/validation/test
  segments. 【F:src/data/walkforward.py†L24-L118】
- **Train-only normalisation.** StandardScaler objects are fit on the training
  slice and applied to validation/test data to avoid leakage. 【F:src/data/walkforward.py†L122-L160】
- **Leak guard.** Timestamp continuity checks prevent feature windows from
  reaching beyond the forecasting horizon; regressions are unit-tested. 【F:src/data/walkforward.py†L162-L218】【F:tests/test_leak.py†L1-L74】
- **Metric regeneration.** Tables and figures are rebuilt directly from stored
  per-run metrics so reviewers can audit published numbers. 【F:scripts/reproduce_all.py†L1-L115】

## Dataset handling
- **Source data.** Synthetic FX candles seeded from anonymised statistics ship in
  `data/sample.csv` for local smoke tests; real studies should replace this with
  institution-approved market data. 【F:data/sample.csv†L1-L10】
- **Timezone harmonisation.** Timestamps are parsed with explicit source and
  target timezones, resolving ambiguous or missing offsets to ensure consistent
  sequencing across global trading sessions. 【F:src/data/walkforward.py†L60-L101】
- **Feature windows.** Sliding sequences respect the configured lookback and
  horizon multiples so that each example contains contiguous market context. 【F:src/data/walkforward.py†L170-L218】

## Fairness, safety, and guardrails
- **No-look-ahead enforcement** eliminates information leakage that could inflate
  scores and disadvantage baseline models. 【F:tests/test_leak.py†L44-L74】
- **Per-pair evaluation** ensures that underrepresented currency pairs are not
  obscured by aggregate metrics; embargo gaps in metadata document how much
  buffering protects against cross-window contamination. 【F:src/data/walkforward.py†L36-L156】
- **Synthetic defaults** avoid exposing proprietary or personally identifiable
  information; researchers must validate licensing and compliance when swapping
  in live data.
- **Responsible-use reminder.** Forecasts are provided for research benchmarking
  and should not be deployed for financial decision-making without additional
  risk controls (see the model card for warnings). 【F:docs/model_card.md†L10-L40】

## Reproducibility and archival
1. Record seeds, hardware manifests, and resolved configs emitted under
   `artifacts/runs/` during multi-run training.
2. Regenerate publication tables and figures with
   `python scripts/reproduce_all.py --manifest artifacts/paper_manifest.json` to
   capture provenance of every metric file. 【F:scripts/reproduce_all.py†L71-L115】
3. Package `paper_outputs/`, curated configs from `artifacts/configs/`, and all
   release scripts when preparing an archival deposit (e.g., Zenodo).
4. Follow the [Release archival workflow](../README.md#release-archival-workflow)
   for DOI minting, upload validation, and README cross-checks.

## Contact
Questions and feedback are welcome via issues or pull requests on the project
repository.
