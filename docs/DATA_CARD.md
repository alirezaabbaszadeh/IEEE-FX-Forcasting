# IEEE FX Forecasting Data Card

## Summary
This data card documents the synthetic fixture dataset bundled with the IEEE FX
Forecasting benchmark and the expectations for substituting production-grade FX
feeds. The fixture provides deterministic inputs for CI and documentation;
research deployments should swap in licensed market feeds that meet compliance
requirements.

## Dataset origin
- **Provenance.** The fixture is generated from anonymised statistical profiles
  of EUR/USD spot prices and stored in `data/sample.csv`.【F:data/sample.csv†L1-L10】
- **Fields.** Each record includes an ISO-8601 timestamp, the trading pair code,
  and OHLC candle values (`Open`, `High`, `Low`, `Close`).【F:data/sample.csv†L1-L10】
- **Licensing.** Synthetic data ships without restrictions; real-market
  replacements must respect vendor terms and jurisdictional limits.

## Collection and preprocessing
- **Loading.** `src.data.dataset.load_dataframe` validates required columns and
  drops rows with missing features or targets before any window construction.
  【F:src/data/dataset.py†L110-L145】
- **Timezone handling.** `src.data.walkforward.WalkForwardSplitter` normalises
  timestamps into an explicit target timezone, handling ambiguous or missing
  offsets to guarantee chronological consistency across windows.【F:src/data/walkforward.py†L60-L118】
- **Pair filtering.** The splitter enforces explicit currency pair selection and
  raises when no rows remain, preventing silent drops of minority pairs.【F:src/data/walkforward.py†L34-L68】
- **Normalization.** StandardScaler instances fit on training slices are reused
  for validation/test partitions to avoid leakage.【F:src/data/walkforward.py†L120-L218】
- **Sequence creation.** Sliding windows respect lookback and horizon multiples,
  ensuring inputs and targets stay aligned without skipping timestamps.【F:src/data/walkforward.py†L170-L218】

## Splitting strategy
- **Walk-forward windows.** Configurable train/validation/test lengths, steps,
  and embargo gaps are enforced through the walk-forward splitter.【F:src/data/walkforward.py†L24-L118】
- **Metadata capture.** Window diagnostics store index ranges, embargo gaps, and
  calendar settings so reviewers can reconstruct the evaluation context.
  【F:src/data/walkforward.py†L134-L156】

## Quality checks and guardrails
- **Leak detection.** Timestamp continuity checks raise when feature windows
  reach beyond the forecasting horizon; regression tests cover the guard.
  【F:src/data/walkforward.py†L162-L218】【F:tests/test_leak.py†L1-L74】
- **Missing data handling.** Required feature/target columns must be present and
  non-null before dataset creation, preventing silent NaN propagation.
  【F:src/data/dataset.py†L118-L138】
- **Calendar transparency.** Timezone and calendar settings accompany each
  window, enabling audits of market session alignment.【F:src/data/walkforward.py†L60-L158】

## Fairness and responsible use
- **Equal treatment across pairs.** Explicit filtering and erroring on missing
  pairs prevents benchmarks from ignoring low-liquidity currencies.【F:src/data/walkforward.py†L34-L68】
- **Synthetic default.** Shipping synthetic data minimises privacy risk and keeps
  the repository safe for public distribution; replace it only with datasets that
  have cleared legal review.
- **Use limitations.** Outputs are for research benchmarking only and should not
  drive financial decisions without additional validation, governance, and risk
  controls.【F:docs/model_card.md†L9-L40】

## Recommended citations
Use the `CITATION.cff` metadata once DOI minting completes; include the dataset
DOI or access statement if an alternative market feed is employed.
