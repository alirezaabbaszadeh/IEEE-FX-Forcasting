# Mix-of-Experts FX Forecasting Model Card

## Model Details
- **Authors:** IEEE FX Forecasting project contributors
- **Architecture:** Residual temporal convolution + attention encoder with a
  Mix-of-Experts transformer head implemented in PyTorch.【F:src/models/moe_transformer.py†L1-L200】
- **Framework:** PyTorch 2.1 (CPU build)
- **License:** MIT

## Intended Use
The model targets short-horizon FX rate forecasting for research and
benchmarking. It is intended for:
- Ablation studies on multi-horizon forecasting benchmarks.
- Demonstrations of interpretability tooling (attention heatmaps, expert
  utilisation, gradient attributions).
- Reproducible baselines for academic publications.

It is **not** intended for high-frequency trading or production deployment
without additional validation.

## Training Data
- **Source:** Synthetic FX spot rate trajectories derived from anonymised
  historical statistics.【F:data/sample.csv†L1-L10】
- **Pre-processing:** Z-score normalisation per split using rolling windows as
  described in the walk-forward pipeline.【F:src/data/walkforward.py†L120-L218】
- **Data Provenance:** Generated fixtures stored in `data/sample.csv`; replace
  with institution-approved datasets for real studies.

## Evaluation Data
- Synthetic validation/test splits created via chronological partitioning with
  configurable embargo gaps.【F:src/data/walkforward.py†L24-L118】
- Metrics reported: MAE, RMSE, sMAPE, MASE, interval coverage, CRPS, and
  interpretability diagnostics captured during evaluation.
  【F:src/metrics/point.py†L1-L72】【F:src/metrics/calibration.py†L1-L120】

## Metrics & Validation
- Core metrics logged per epoch in the structured run artifacts under
  `artifacts/runs/.../metrics.json` and aggregated via `src.analysis.stats`.
  【F:scripts/reproduce_all.py†L110-L180】【F:src/analysis/stats.py†L1-L220】
- Statistical significance testing available via `src.analysis.stats` utilities.
- Interpretability artefacts (attention, attributions) generated through
  `python -m src.analysis.interpretability` or the CLI hooks integrated into the
  reproduction script.【F:src/analysis/interpretability.py†L1-L210】

## Reproducibility
1. Run `./scripts/reproduce_all.sh` from the repository root. The script
   provisions the Conda environment, executes the multi-run training workflow,
   triggers calibration/statistics CLIs, and rebuilds `paper_outputs/` from the
   generated artifacts.【F:scripts/reproduce_all.sh†L1-L66】
2. For a lightweight smoke validation, execute
   `./scripts/reproduce_all.sh --smoke --no-conda`; this mirrors the CI regression
   guard and keeps runtimes short.
3. The script wipes `artifacts/` and `paper_outputs/` before starting and fails
   if required predictions or Diebold-Mariano caches are missing, ensuring the
   release artifacts are produced from a clean slate.【F:scripts/reproduce_all.py†L71-L180】

Hydra training runs materialise under
`artifacts/runs/<model>/<config_hash>/<pair>_<horizon>/window-*/`. Each window
includes:
- `metrics.json` summarising the final training/validation losses.
- `metadata.json` capturing the resolved config hash, dataset checksums,
  environment lockfile digests, and an artifact index.
- Optional `benchmarks/` outputs containing CSV/JSON latency measurements.

Resolved configs are deduplicated beneath `artifacts/configs/<hash>.yaml`,
allowing the reproduction script to reuse the exact Hydra payload. The manifest
emitted by `scripts/reproduce_all.py` records metrics sources, regenerated
figures, and referenced config snapshots to support full provenance when
publishing results.【F:scripts/reproduce_all.py†L86-L180】

## Ethical Considerations & Risks
- Synthetic fixtures lack real-world risk factors; deploying on live data
  requires rigorous compliance checks.
- FX forecasting may influence financial decisions; misuse can amplify market
  volatility.
- Interpretability visualisations provide diagnostic cues but should not be
  treated as causal explanations.【F:src/analysis/interpretability.py†L1-L210】

## Deployment & Monitoring
- Use the benchmarking utilities in `scripts/benchmark.py` to validate latency
  and memory before deployment.【F:scripts/benchmark.py†L1-L180】
- Monitor drift by periodically recomputing attribution summaries and comparing
  against historical baselines using the interpretability toolkit.

## Contact & Feedback
Please open an issue or submit a pull request on the project repository for
questions, bug reports, or collaboration proposals.
