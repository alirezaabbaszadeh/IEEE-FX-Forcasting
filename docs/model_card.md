# Mix-of-Experts FX Forecasting Model Card

## Model Details
- **Authors:** IEEE FX Forecasting project contributors
- **Architecture:** Residual temporal convolution + attention encoder with a Mix-of-Experts transformer head
- **Framework:** PyTorch 2.1 (CPU build)
- **License:** MIT

## Intended Use
The model targets short-horizon FX rate forecasting for research and benchmarking. It is intended for:
- Ablation studies on multi-horizon forecasting benchmarks.
- Demonstrations of interpretability tooling (attention heatmaps, expert utilisation, gradient attributions).
- Reproducible baselines for academic publications.

It is **not** intended for high-frequency trading or production deployment without additional validation.

## Training Data
- **Source:** Synthetic FX spot rate trajectories derived from anonymised historical statistics.
- **Pre-processing:** Z-score normalisation per split using rolling windows.
- **Data Provenance:** Generated fixtures stored in `data/sample.csv`; replace with institution-approved datasets for real studies.

## Evaluation Data
- Synthetic validation/test splits created via chronological partitioning.
- Metrics reported: mean absolute error (MAE), mean squared error (MSE), directional accuracy, attention/expert diagnostics.

## Metrics & Validation
- Core metrics logged per epoch in `artifacts/examples/metrics.csv`.
- Statistical significance testing available via `src.analysis.stats` utilities.
- Interpretability artefacts (attention, attributions) generated through `python -m src.analysis.interpretability`.

## Reproducibility
1. Create the Conda environment: `conda env create -f environment.yml`
2. Activate it: `conda activate ieee-fx`
3. Run a smoke training job: `python -m src.cli training.epochs=1 training.device=cpu`
4. Publish analysis artefacts: `make publish`

These steps regenerate tables, figures, and packaged archives under `artifacts/`.

## Ethical Considerations & Risks
- Synthetic fixtures lack real-world risk factors; deploying on live data requires rigorous compliance checks.
- FX forecasting may influence financial decisions; misuse can amplify market volatility.
- Interpretability visualisations provide diagnostic cues but should not be treated as causal explanations.

## Deployment & Monitoring
- Use the benchmarking utilities in `src.analysis.benchmark` to validate latency and memory before deployment.
- Monitor drift by periodically recomputing attribution summaries and comparing against historical baselines.

## Contact & Feedback
Please open an issue or submit a pull request on the project repository for questions, bug reports, or collaboration proposals.
