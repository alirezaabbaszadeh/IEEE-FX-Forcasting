# IEEE FX Forecasting

The IEEE FX Forecasting project packages the benchmark, research code, and
publication assets that accompany our probabilistic mix-of-experts FX model. The
repository consolidates the production-ready implementation under `src/` while
retaining historical `v_*` prototypes for provenance. This document orients
contributors and reviewers; focused documentation lives in the cards under
`docs/`.

## Quick start

```bash
python -m pip install -r requirements-dev.txt
make ci
```

The requirements file installs the development dependencies, and `make ci` mirrors the
repository smoke CI by running linting, type checks, unit tests, and the
deterministic training regression against the synthetic fixture dataset in
`data/sample.csv`. Use `python -m src.cli ...` to access the Hydra-powered command
line interface when triggering bespoke jobs.【F:pyproject.toml†L1-L49】【F:Makefile†L1-L64】

For an end-to-end rebuild of tables, figures, and manifests, run:

```bash
./scripts/reproduce_all.sh --smoke
```

Drop the `--smoke` flag to execute the full multi-seed workflow when preparing a
release. The script optionally provisions Conda (pass `--no-conda` in
containerised environments) and clears existing `artifacts/` and
`paper_outputs/` directories to guarantee reproducibility.【F:scripts/reproduce_all.sh†L1-L66】

## Key capabilities

- **Deterministic multirun training.** `src.training.runner` coordinates multi-
  seed experiments, sets deterministic flags, and writes manifests with seed,
  git commit, hardware, and resolved configuration hashes for audit-ready
  reporting.【F:src/training/runner.py†L1-L200】
- **Time-series correct data pipeline.**
  `src.data.walkforward.WalkForwardSplitter` enforces timezone normalisation,
  embargoed walk-forward splits, and leakage guards inherited from the
  historical prototypes without duplicating utilities.【F:src/data/walkforward.py†L1-L218】
- **Uncertainty-aware inference.** The purged conformal calibration stack ships
  with governance defaults in `configs/inference/pcc.yaml` and evaluation
  routines in `src.inference.conformal_purged`, keeping coverage claims
  reproducible and embargo compliant.【F:configs/inference/pcc.yaml†L1-L8】【F:src/inference/conformal_purged.py†L1-L200】
- **Statistical reporting.** `scripts/export_tables.py` and
  `scripts/export_figures.py` rebuild publication assets directly from structured
  metrics, while `src.analysis.stats` implements Diebold-Mariano, ANOVA/Welch,
  and SPA/MCS guards referenced by the claim freeze documentation.【F:scripts/export_tables.py†L1-L160】【F:src/analysis/stats.py†L1-L220】

## Repository layout

- `src/` — unified research codebase spanning data, training, inference,
  analysis, and reporting.
- `configs/` — Hydra schemas and governance manifests, including the PCC freeze
  record consumed by the publication pipeline.
- `scripts/` — automation entry points for training, calibration, benchmarking,
  table/figure exports, and archival packaging.
- `docs/` — benchmark, data, model, developer, and research cards (see
  Documentation map).
- `paper_outputs/` — regenerated figures, tables, and ablation summaries keyed to
  reproduction manifests.
- `v_*` — archived prototype implementations retained for provenance; use the
  consolidated modules in `src/` for new work.【F:src/__init__.py†L1-L30】

## Developer workflow

1. `python -m pip install -r requirements-dev.txt` or `conda env create -f
   environment.yml` to reproduce the locked environment.
2. Use `python -m src.cli` for ad-hoc single runs or `python -m src.cli --multirun`
   for multi-seed experiments; invoke `make train-smoke` for the deterministic
   regression suite.
3. Run `pytest` for targeted tests; `pytest tests/test_leak.py` isolates the
   timestamp continuity guard when modifying walk-forward logic.【F:pytest.ini†L1-L9】【F:tests/test_leak.py†L1-L74】
4. Capture new publication assets with `python scripts/export_tables.py` and
   `python scripts/export_figures.py` after populating `artifacts/` via the
   training workflow.

Structured outputs follow the
`artifacts/runs/<model>/<config_hash>/<pair>_<horizon>/<window>/seed-<id>/`
layout, with aggregates under `artifacts/aggregates/`. Metadata files include
resolved configs, environment snapshots, and compute manifests for audit
trails.【F:scripts/reproduce_all.py†L1-L180】

## Documentation map

- `docs/BENCHMARK_CARD.md` — evaluation protocol, fairness commitments, and
  guardrails.
- `docs/DATA_CARD.md` — dataset provenance, preprocessing controls, and leakage
  defences.
- `docs/model_card.md` — architecture summary, intended use, risks, and
  reproducibility recipe.
- `docs/developer.md` — leak guard remediation, governance reminders, and
  day-to-day contributor workflow.
- `docs/pcc_claim_freeze.md` — governance record for calibrated interval claims.
- `docs/refactor_plan.md` & `docs/research_plan.md` — roadmap and active research
  funnel.
- `docs/REPOSITORY_CLEANUP.md` — ledger describing the archived TensorFlow
  prototypes (`v_01` … `v_10`) and the hygiene expectations when touching them.

Each card links to the modules enforcing the guarantees so reviewers can trace
claims to concrete code.

## Research execution standards

The benchmark continues to enforce the nine execution pillars required for
publishable FX forecasting research. Automation across the repository satisfies
these expectations:

1. **Deterministic multirun module** capturing seeds, device state, and metadata
   across ≥5 runs for major baselines.【F:src/training/runner.py†L40-L132】
2. **Multi-pair, multi-horizon walk-forward evaluation** with embargo gaps and
   session-aware diagnostics.【F:src/data/walkforward.py†L24-L218】
3. **Hyperparameter sensitivity** via Sobol/Bayesian search harnesses that score
   the mean metric across runs and persist top-k configs.【F:src/analysis/hparam.py†L1-L210】
4. **Statistical testing** covering ANOVA/Welch, Tukey/Dunn, and Diebold-Mariano
   with Newey-West adjustments.【F:src/analysis/stats.py†L1-L220】
5. **Interpretability reporting** including attention heatmaps, expert
   utilisation timelines, and gradient-based sanity checks via
   `src.analysis.interpretability`.【F:src/analysis/interpretability.py†L1-L210】
6. **Compute benchmarking** implemented in `scripts/benchmark.py`, recording
   throughput, latency, and memory alongside configuration manifests.【F:scripts/benchmark.py†L1-L180】
7. **Publishable artifacts** produced by the export scripts and governed by the
   manifest indices stored in `artifacts/` and `paper_outputs/`.
8. **Single-source configuration** enforced by Hydra schemas that fingerprint
   each resolved run.【F:configs/schema/model.yaml†L1-L120】
9. **Tests and lightweight CI** delivered via the Makefile targets noted in the
   quick start section.【F:Makefile†L1-L64】

The companion “Gold Standard” requirements expand on reproducibility,
uncertainty, economic realism, compute governance, transparent data handling,
documentation, statistical integrity, and ethics. Detailed expectations live in
the benchmark, data, and model cards plus the governance profiles under
`configs/governance/`.

## Release checklist

1. Regenerate publication assets with
   `python scripts/reproduce_all.py --manifest artifacts/paper_manifest.json` to
   record metrics sources, resolved configs, and aggregate outputs.
2. **Freeze configurations.** Copy referenced configs from `artifacts/configs/`
   and the CLI entry scripts under `scripts/` into a `release_bundle/` directory
   alongside `paper_outputs/`.
3. **Verify datasets.** Confirm dataset checksums and manifest timestamps align
   with the release candidate; rerun `make train-smoke` after pulling fresh
   fixtures to catch stale caches.
4. **Audit compute manifests.** Inspect `artifacts/*/compute.json` for driver
   versions, precision modes, and hardware drift since the previous release.
5. **Mint or update the DOI.** Record the minted DOI in `CITATION.cff` and the
   archival metadata once the upload succeeds.
6. **Cross-check documentation.** Ensure the README, benchmark card, and model
   card reflect the release tag, DOI, and claim freeze metadata.
7. **Create the archival record.** Package the regenerated `paper_outputs/`, the
   curated configs from `artifacts/configs/`, and the release manifest for
   upload to the archival service. Include the generated manifest JSON so
   reviewers can retrace the files underlying published tables.

See `docs/BENCHMARK_CARD.md` for a narrative summary of the release workflow and
responsible-use guidance.
