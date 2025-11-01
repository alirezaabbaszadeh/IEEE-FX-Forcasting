# Refactor Roadmap for Original Research 2026

This roadmap breaks the refactor into phases aligned with the new research expectations. Each phase contains a single flagship task that summarizes the workstream and highlights the concrete sub-components required.

## Phase 1 — Establish deterministic research scaffold
**Task: Build reproducible core in new `src/` platform**
- Consolidate `v_*` prototypes into a unified package structure under `src/`, preserving historically validated components while eliminating duplicate utilities.
- Introduce containerized and conda-based environments with pinned dependencies (`Dockerfile`, `environment.yml`, `pyproject.toml`) and document one-command bootstrap scripts.
- Implement global seed management utilities covering Python, NumPy, torch, CUDA/cuDNN, and dataloader workers; enforce deterministic kernels where available.
- Add configuration management (YAML + schema validation) that records resolved configs, git commit hashes, environment fingerprints, and dataset checksums into every run artifact.
- Scaffold logging hooks that persist metadata (`run_id`, hardware, driver, precision mode) and route metrics to structured artifact directories.

## Phase 2 — Data ingestion and walk-forward evaluation
**Task: Implement time-series correct data & evaluation pipelines**
- Build modular data loaders for multi-pair and multi-horizon FX datasets with timezone normalization and trading-calendar awareness, ensuring missing data, DST transitions, and resampling policies are explicit.
- Enforce train-only normalization statistics and leakage-safe feature engineering; cache normalization fingerprints for reuse across evaluation slices.
- Implement embargoed walk-forward split generators supporting rolling and expanding windows, with validation/test boundaries derived strictly from chronological order.
- Provide evaluation runners that compute pair×horizon metrics, macro/micro aggregates, and stratified summaries by volatility regime and session.
- Integrate baseline models to verify identical data handling, enabling regression tests against existing `v_*` results.

## Phase 3 — Modeling, hyperparameter search, and statistical validation
**Task: Build experiment engine with search and stats**
- Design a run orchestrator that executes multi-seed experiments, collates metrics (mean, std, 95% CI), and archives per-run metadata artifacts.
- Implement configurable hyperparameter search backends (Sobol / Bayesian) with budgets, early stopping, and multi-fidelity support; ensure objective functions average metrics across seeds.
- Capture top-k configurations, sensitivity analyses, and partial dependence plots; persist search traces for reproducibility.
- Add statistical testing suite covering ANOVA/Welch/Kruskal-Wallis, Diebold-Mariano with Newey-West adjustments, and multiple-comparison controls (Tukey HSD, Holm, SPA/MCS).
- Automate reporting of effect sizes, power estimates, and sanity checks for statistical assumptions.

## Phase 4 — Interpretability, compute benchmarking, and publication artifacts
**Task: Produce explainability & artifact pipeline**
- Implement attention and mixture-of-experts interpretability tooling: heatmaps, expert-utilization timelines, gating entropy, gradient-based attributions, and perturbation sanity checks.
- Build compute benchmarking harnesses measuring training/inference throughput, latency percentiles, memory footprint, and parameter counts under standardized hardware settings.
- Generate publication-ready figures (vector graphics, 300 dpi PNGs) and tables with consistent styling, alongside machine-readable `metadata.json` and structured CSV/Parquet outputs.
- Assemble archival packages including run logs, resolved configs, environment manifests, and reproduction scripts; integrate checksum verification and persistent identifiers.
- Draft model cards and documentation updates summarizing intended use, limitations, risks, and compliance considerations.
