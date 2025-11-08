# Refactor Roadmap for Original Research 2026

This roadmap tracks consolidation progress and upcoming engineering work aligned
with the research expectations documented in the benchmark and model cards.
Statuses reflect the current `src/` implementation and highlight remaining gaps.

## Phase 1 — Deterministic research scaffold *(Status: Complete)*
**Goal: Build a reproducible core in the unified `src/` platform.**
- ✅ Consolidated the historical `v_*` prototypes into the modular package
  structure now exposed under `src/`, preserving validated components while
  removing duplicated utilities.【F:src/__init__.py†L1-L30】
- ✅ Introduced containerised and Conda environments with pinned dependencies via
  `Dockerfile`, `environment.yml`, and `pyproject.toml`, plus Make targets for
  one-command bootstrap scripts.【F:Dockerfile†L1-L80】【F:pyproject.toml†L1-L49】
- ✅ Implemented global seed management utilities covering Python, NumPy, torch,
  CUDA/cuDNN, and dataloader workers within `src.training.runner` and
  `src.utils.repro`.【F:src/training/runner.py†L1-L200】【F:src/utils/repro.py†L1-L160】
- ✅ Added Hydra configuration management that records resolved configs, git
  commits, environment fingerprints, and dataset checksums into every run
  artifact via `src.utils.manifest` and the reproduction script.
  【F:src/utils/manifest.py†L1-L200】【F:scripts/reproduce_all.py†L86-L180】

## Phase 2 — Data ingestion and walk-forward evaluation *(Status: Complete)*
**Goal: Maintain time-series correct data & evaluation pipelines.**
- ✅ Modular data loaders for multi-pair FX datasets with timezone normalisation
  and trading-calendar hooks live in `src.data.dataset` and
  `src.data.walkforward`.【F:src/data/dataset.py†L1-L160】【F:src/data/walkforward.py†L1-L218】
- ✅ Train-only normalisation and leakage-safe feature engineering are enforced by
  `WalkForwardSplitter`, with scaler fingerprints persisted in metadata.
  【F:src/data/walkforward.py†L118-L218】
- ✅ Embargoed walk-forward split generators supporting rolling windows reside in
  `src.splits.walk_forward`; evaluation runners compute per pair × horizon
  metrics with stratified summaries in `src.analysis.benchmark`.
  【F:src/splits/walk_forward.py†L1-L200】【F:src/analysis/benchmark.py†L1-L220】
- 🔄 TODO: Extend calendar integration with explicit market-holiday libraries and
  expose CLI overrides for trading sessions.

## Phase 3 — Modeling, hyperparameter search, and statistical validation *(Status: In progress)*
**Goal: Mature the experiment engine with search and statistical guards.**
- ✅ Multi-seed orchestration archives per-run manifests, aggregates mean/std/CI,
  and logs hardware metadata via `src.training.runner`.
- ✅ Hyperparameter search harnesses using Sobol and Bayesian optimisation live in
  `src.analysis.hparam`, persisting search traces and top-k configs.
  【F:src/analysis/hparam.py†L1-L210】
- ✅ Statistical testing suite covering ANOVA/Welch, Tukey/Dunn, Diebold-Mariano,
  and SPA/MCS is implemented in `src.analysis.stats` and wired into the
  reproduction script.
- 🔄 TODO: Add partial dependence plots and rank consistency dashboards under
  `src.reporting` to visualise sensitivity analyses.

## Phase 4 — Interpretability, compute benchmarking, and publication artifacts *(Status: In progress)*
**Goal: Produce explainability & artifact pipelines ready for archival releases.**
- ✅ Attention and MoE interpretability tooling (heatmaps, expert utilisation,
  gating entropy, gradient attributions) ships in
  `src.analysis.interpretability` with CLI access.【F:src/analysis/interpretability.py†L1-L210】
- ✅ Compute benchmarking harnesses measuring throughput, latency, and memory are
  bundled in `scripts/benchmark.py` and integrated with metadata manifests.
  【F:scripts/benchmark.py†L1-L180】
- ✅ Publication-ready tables/figures with consistent styling are regenerated via
  `scripts/export_tables.py`, `scripts/export_figures.py`, and the manifest-aware
  reproduction pipeline.
- 🔄 TODO: Automate figure metadata validation and enforce consistent typography
  across new assets added to `paper_outputs/`.
- 🔄 TODO: Expand the documentation changelog to summarise claim updates per
  release tag in coordination with `docs/pcc_claim_freeze.md`.
