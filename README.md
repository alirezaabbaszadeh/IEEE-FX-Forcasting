**Executable Code Workplan — Explanatory Version (with tightened standards)**

No code snippets, no screenshots. This is the operational spec you implement so the paper reads like original research, not a class project.

## 1) Multi-run module with controlled seeds

* **Goal:** Stable performance estimates that aren’t artifacts of lucky initialization or a “good GPU day.”
* **What to include:** Full control of all RNG sources (framework, NumPy, Python, CUDA/cuDNN, dataloader workers), deterministic kernels where available, and explicit logging of `seed`, `git commit`, device specs, driver versions, and mixed-precision settings. Aggregate metrics with mean, standard deviation, and 95% CIs.
* **Execution standard:** ≥5 independent runs for Model V8 and key baselines; compute CIs via Student-t and optionally nonparametric bootstrap (1k resamples). Report effect sizes (e.g., Cohen’s d) against the best baseline. Each run writes a standalone artifact with `run_id` and complete metadata.

## 2) Multi-pair / multi-horizon infrastructure with walk-forward evaluation

* **Goal:** Demonstrate generalization beyond a single pair and horizon with deployment-realistic timing.
* **What to include:** A data loader that accepts lists of currency pairs and horizons, timezone normalization, trading-calendar awareness, robust resampling/aggregation, and strict no-look-ahead. Normalization stats computed on train only and applied to val/test. Missing data and DST handled explicitly.
* **Execution standard:** Rolling or expanding walk-forward splits with an **embargo** between train and test to prevent overlap leakage; hyperparameters tuned only on train/val for the current slice. Report pair×horizon results, macro/micro averages, and stratified summaries by volatility regime and session (e.g., London/NY).

## 3) Hyperparameter sensitivity (compact grid/Bayesian optimization)

* **Goal:** Show the model isn’t propped up by a single “magic” setting.
* **What to include:** A declared search budget; principled spaces for the dominant knobs; Sobol or lightweight Bayesian optimization; early-stopping and multi-fidelity options (e.g., ASHA/Hyperband). Plot response curves/partial dependence for key parameters.
* **Execution standard:** Optimization objective is the **mean** metric across multi-run repeats, not a single run. Publish the top-k configs with seeds, and provide stability analyses and rank consistency across splits and pairs.

## 4) Statistical testing

* **Goal:** Separate real differences from noise.
* **What to include:** One-way ANOVA across model variants with Tukey HSD (or comparable) for post-hoc pairwise tests; for forecast errors over time, Diebold-Mariano with Newey-West covariance.
* **Execution standard:** Check assumptions (normality, homoscedasticity). If violated: Welch’s ANOVA or Kruskal-Wallis, with Dunn’s test + Holm correction. Always report p-values, 95% CIs, and effect sizes (η²/ω² or rank-based). Prefer advanced multiple-comparison controls such as MCS or SPA for model sets. Include a brief power analysis or observed power.

## 5) Interpretability (Attention and MoE)

* **Goal:** Explain what the model attends to and when different experts activate.
* **What to include:** Attention heatmaps for specific market events, expert-utilization time series, gating entropy, and correlations with volatility or spread. Include representative success and failure cases.
* **Execution standard:** Use fixed seeds for interpretability extraction, sanity checks under small input perturbations, and printable summaries. Where feasible, complement attention with gradient-based attributions (e.g., IG) to avoid over-interpreting softmax weights.

## 6) Compute benchmarking

* **Goal:** Make the cost of use explicit.
* **What to include:** Per-epoch training time, per-sample and per-batch inference latency, throughput, peak memory, and parameter count.
* **Execution standard:** Short warm-up, then three or more measurements; report mean and stdev, plus p50/p90/p99 latencies for inference. Specify exact hardware, drivers, and precision. Compare all variants under the same batch size and mixed-precision policy.

## 7) Clean, publishable outputs

* **Goal:** Traceable, reusable results without digging through logs.
* **What to include:** Structured CSV/Parquet with stable naming; vector graphics (PDF/SVG) or 300 dpi PNG; consistent fonts, symbols, and units; visible confidence bands.
* **Execution standard:** Conventions such as
  `runs/<model>/<pair>_<horizon>/seed-<id>/metrics.csv`
  Figures have titles, legends, axis labels with units, and CI shading. Include a machine-readable `metadata.json` alongside each table/figure.

## 8) Single-source configuration

* **Goal:** Change scenarios without editing code.
* **What to include:** One YAML (or equivalent) for data paths, pairs, horizons, features, architecture, training, logging, and evaluation toggles.
* **Execution standard:** Schema validation, CLI overrides with precedence, config fingerprint hashed into outputs, environment lockfile recorded, and dataset checksums verified at load. Store the resolved config next to each run.

## 9) Tests and lightweight CI

* **Goal:** Prevent silent breakage during refactors.
* **What to include:** Unit tests for data loaders and metrics, leakage tests, reproducibility checks, and a smoke-test training loop on a tiny dataset.
* **Execution standard:** CI runs on a minimal fixture dataset; lint/format/type checks; coverage reported; artifacts from the smoke run uploaded. CI fails if reproducibility error exceeds a preset tolerance.

---

# “Gold Standard” Requirements for Original Research

## A. Reproducibility and provenance

* Environment captured via container or conda with pinned versions; deterministic flags documented; `git commit` embedded in logs and outputs; a one-command script to regenerate all tables and figures end-to-end.
* All random seeds specified; hardware and driver versions logged; exact data snapshot (with checksum) archived or programmatically re-created.

## B. Time-series-correct evaluation

* No look-ahead, proper walk-forward with embargo, and hyperparameter tuning restricted to in-slice train/val only.
* Baselines both classical and modern, evaluated under identical data splits, normalization protocols, and compute budgets.
* Distribution shift checks: performance stratified by volatility, sessions, crises, and structural break periods; optional change-point tests.

## C. Uncertainty and calibration

* Report predictive intervals (e.g., conformal or quantile models), empirical coverage vs nominal, and calibration plots. Include log-loss or CRPS where applicable.

## D. Economic realism (if claiming practical forecasting value)

* Backtests that include transaction costs, slippage, and execution latency; position sizing and turnover; risk metrics (drawdown, Sharpe, Sortino) with CIs. Separate research claims from trading claims if you are not providing a full execution model.

## E. Compute governance

* Declare a compute budget; align budgets across models; report “improvement per unit cost” (e.g., RMSE reduction per GPU-hour). Include energy estimate if feasible.

## F. Transparent data handling

* Data sources, licenses, and any filtering or revisions documented. Preprocessing steps, missing-value policy, outlier handling, and resampling rules specified. Ensure train-only normalization and feature engineering.

## G. Documentation and artifacts

* Release code, configs, and a minimal working dataset or data-rebuild scripts under a permissive license consistent with data terms. Provide a Model Card covering intended use, limitations, risks, and failure modes.
* Archive artifacts with a persistent identifier; include `CITATION.cff`, README with exact reproduction commands, and a changelog.

## H. Statistical integrity

* Multiple-comparison control across models and horizons; effect sizes alongside p-values; interval estimates emphasized over point estimates. Where feasible, include SPA/MCS and robustness to alternative metrics.

## I. Security, compliance, and ethics

* No PII; secure secrets and API keys; document any external services. Clarify non-advisory nature of outputs and known risks of deployment in high-stakes settings.

---

Implementing this framework removes the “maybe you got lucky” objection. Even if luck helped, the reporting standards here make luck irrelevant.
