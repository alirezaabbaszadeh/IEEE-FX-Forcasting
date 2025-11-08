# Repository cleanup ledger

## Purpose
This ledger captures the state of the consolidated IEEE FX Forecasting repository
and the historical TensorFlow pipelines retained for provenance. It clarifies
which directories are active, which are archived, and the hygiene steps required
after touching either side of the tree.

## Unified `src/` platform (active)
- PyTorch research stack with the mix-of-experts transformer, conformal
  calibration utilities, and reproducibility helpers. Key entry points live in
  `src/cli.py`, `src/models/moe_transformer.py`, and
  `src/inference/conformal_purged.py`.
- Shared tooling (`scripts/reproduce_all.sh`, `scripts/export_tables.py`,
  `scripts/export_figures.py`) regenerates training runs, calibration artefacts,
  and publication assets from the manifests produced under `artifacts/`.
- Modern development workflow: editable install via `pyproject.toml`, linting
  and typing gates wired through `Makefile` targets, and documentation cards in
  `docs/` for benchmark, data, model, governance, and research plans.

## Archived `v_*` prototypes (read-only)
- Ten directories (`v_01` … `v_10`) snapshot the original TensorFlow pipelines
  used before the PyTorch refactor. Each bundle includes orchestration scripts
  (`Run.py`, `MainClass.py`) plus helper modules (`DataLoader.py`,
  `ModelBuilder.py`, `Trainer.py`, `Evaluator.py`, `HistoryManager.py`,
  `ModelManager.py`, `VisualPredictions.py`) and default hyperparameter YAMLs.
- Pipelines import TensorFlow/Keras directly and configure the environment with
  manual session resets, mixed-precision toggles, and CLI-driven defaults.
  Version 1 demonstrates the pattern in `v_01/MainClass.py` and `v_01/Run.py`,
  while Version 10 shows the final iteration with richer CLI arguments and
  logging in `v_10/Run.py`.
- The prototypes depend on CSV files expected alongside each directory. The
  cleaned top-level repository omits those large assets; keep the folders intact
  for historical review but avoid executing them inside CI unless dedicated test
  data is restored.

## When work touches the archives
1. Mirror behaviour back into `src/` so that active code paths stay aligned. If a
   bug fix or experiment only lands in a prototype, document the rationale here
   and in `docs/refactor_plan.md`.
2. Record environment differences. TensorFlow-specific requirements belong in a
   per-version `requirements.txt` or an inline comment inside the prototype.
3. Add a short changelog entry below noting the modification, the motivation, and
   whether parity tests were run against the PyTorch implementation.

### Archive changelog
- *2024-XX-XX:* … (add entries here when prototypes change).

## Pre-release hygiene checklist
- ✅ Confirm `src/` and `configs/` contain the authoritative implementations for
  any assets promoted in documentation or publications.
- ✅ Ensure `scripts/reproduce_all.sh` completes from a clean checkout so that
  regenerated tables/figures do not rely on TensorFlow prototypes.
- ✅ Verify `docs/` references the consolidated stack and that this ledger remains
  accurate when prototype folders are added, renamed, or retired.
