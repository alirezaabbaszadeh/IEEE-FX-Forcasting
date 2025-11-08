# Developer guide

## Environment and tooling

- Create a development environment with `pip install -e .[dev]` or
  `conda env create -f environment.yml`; both pin the dependencies used by CI.
- Run `make ci` before pushing to ensure linting, static typing, unit tests, and
  the deterministic smoke training pipeline stay green.【F:Makefile†L1-L64】
- Use `pre-commit run --all-files` when touching formatting-sensitive files; the
  hooks mirror CI defaults for isort, black, and ruff settings defined in
  `pyproject.toml`.【F:pyproject.toml†L1-L49】

## Workflow expectations

1. Launch experiments via `python -m src.cli train` or the pre-baked Make targets
   in the repository root. CLI invocations resolve Hydra configs from `configs/`
   and emit manifests under `artifacts/` for every seed.【F:src/cli.py†L1-L200】
2. Check new metrics with the reporting helpers in `src.reporting` and regenerate
   publication tables/figures using `python scripts/export_tables.py` and
   `python scripts/export_figures.py` after the training jobs finish.【F:scripts/export_tables.py†L1-L160】【F:scripts/export_figures.py†L1-L160】
3. When modifying Hydra schemas, update the accompanying documentation in
   `docs/` and confirm the resolved config fingerprints stored alongside run
   artifacts still match the expectations in governance manifests such as
   `configs/governance/claim_freeze.yaml`.【F:configs/governance/claim_freeze.yaml†L1-L7】

## Leak guard remediation

The walk-forward dataset builder validates that each feature window stays at
least one forecasting horizon ahead of its target timestamp. When this guard
raises a `ValueError` during `prepare_datasets`, use the following checklist to
remediate the issue:

- Inspect the offending split named in the error to confirm whether raw
  timestamps bunch up within the forecast horizon (for example, duplicate ticks
  or mismatched resampling).
- Realign feature columns with the correct timestamps before exporting CSVs.
  Avoid shifting features forward in time or mixing measurements from multiple
  clocks.
- Remove or aggregate irregular observations that compress the gap below the
  configured horizon.
- If the data is legitimately denser than the requested horizon, update the
  forecasting horizon or the model lookback so that each target remains strictly
  after the final feature timestamp.

Run `pytest tests/test_leak.py` locally to reproduce the failure after applying
fixes. The leak guard is enforced automatically by the CI workflows, so merges
are blocked until the test passes.【F:tests/test_leak.py†L1-L74】

## Governance reminders

- The purged conformal calibration (PCC) claim freeze locks the calibrated
  profile in `configs/inference/pcc.yaml`; changes to calibration routines must
  demonstrate ≥2% improvement in CRPS or coverage error as noted in
  `docs/pcc_claim_freeze.md`.【F:configs/inference/pcc.yaml†L1-L8】【F:docs/pcc_claim_freeze.md†L1-L32】
- Record new research claims in `paper_outputs/` by updating the manifests
  produced via `python scripts/reproduce_all.py --manifest ...`; this guarantees
  provenance for every published table or figure.【F:scripts/reproduce_all.py†L1-L180】
- When touching legacy prototypes under `v_*`, mirror the behaviour inside
  `src/` and update `docs/refactor_plan.md` if the consolidation roadmap changes.

## Prototype archives

- Treat `v_01` … `v_10` as read-only TensorFlow snapshots. Their orchestration
  scripts (`Run.py`, `MainClass.py`) and helpers live beside the defaults under
  each directory and rely on local CSV assets that are no longer bundled in the
  cleaned root repository.【F:v_01/MainClass.py†L1-L120】【F:v_10/Run.py†L1-L120】
- Document any change to a prototype in `docs/REPOSITORY_CLEANUP.md` so the
  archive ledger and active `src/` implementation stay consistent. Mirror fixes
  back into the PyTorch stack before merging so CI continues to rely solely on
  the consolidated code path.【F:docs/REPOSITORY_CLEANUP.md†L1-L69】
