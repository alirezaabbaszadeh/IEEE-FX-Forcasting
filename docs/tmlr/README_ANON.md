# Anonymous Artifact Instructions

## Overview
This package recreates every figure and table referenced in the TMLR submission using the
lightweight fixture dataset stored in `data/sample.csv`. The entry point is
`scripts/reproduce_all.sh`, which provisions dependencies, launches the multi-run training
workflow, evaluates the resulting predictions, and regenerates publication assets under
`paper_outputs/`.

## System requirements
- Ubuntu 22.04 LTS (or a comparable POSIX environment with Bash 5.x)
- Python 3.10 with `pip`
- 4 CPU cores and 8 GiB RAM (CPU-only execution)
- ~6 GiB free disk space for intermediate artifacts

> **Note:** GPU acceleration is not required. The included Conda specification installs the
> CPU-only build of PyTorch.

## Preparation
1. Unpack `artifact_anonymous.zip` into a clean working directory.
2. Optionally create a dedicated Conda environment (recommended for isolation):
   ```bash
   conda env create -f environment.yml
   conda activate ieee-fx
   python -m pip install -e .
   ```
   If Conda is unavailable, create a Python virtual environment instead:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements-dev.txt
   python -m pip install -e .
   ```

## End-to-end reproduction
From the artifact root, run:
```bash
./scripts/reproduce_all.sh --no-conda
```
The script performs the following steps:
1. Installs the package in editable mode (skipped when using the Conda workflow above).
2. Removes any existing `artifacts/` and `paper_outputs/` directories.
3. Trains every configuration defined in `configs/default.yaml` via `python -m src.cli --multirun`.
4. Evaluates predictions with conformal calibration and Diebold-Mariano style statistical tests.
5. Collates aggregate CSVs, regenerates publication tables/figures, and writes a manifest to
   `paper_outputs/paper_manifest.json`.

The full run completes in approximately 35 minutes on a 4-core CPU machine. Progress and timing
information are streamed to STDERR with a `[reproduce_all]` prefix.

### Smoke test
Use the optional `--smoke` flag to validate setup quickly (≈2 minutes) without producing the
publication-quality assets. This mode does **not** regenerate the tables and figures cited in the
paper and should only be used for sanity checks.

## Expected outputs
Successful execution recreates the following directories:
- `artifacts/` — per-run metrics, metadata, and aggregate CSVs for each pair/horizon split.
- `paper_outputs/figures/` — vector graphics (PDF) for every figure cited in the submission.
- `paper_outputs/tables/` — CSV exports matching the manuscript tables.
- `paper_outputs/<run_id>_manifest.json` — machine-readable manifest enumerating all inputs and
  outputs, including the resolved Hydra configs.

Use the manifest to confirm that every figure and table aligns with the manuscript identifiers.
Each entry records the source metrics file, aggregate summary, and resolved configuration hash.

## Clean-environment verification checklist
- [ ] Remove any pre-existing `artifacts/` and `paper_outputs/` directories.
- [ ] Create a fresh Conda or virtual environment and install dependencies as above.
- [ ] Run `./scripts/reproduce_all.sh --no-conda` from the artifact root.
- [ ] Confirm that `paper_outputs/tables/` and `paper_outputs/figures/` match the manuscript.
- [ ] Archive `artifact_anonymous.zip` together with the regenerated outputs for review.

## Troubleshooting
- **Conda not available:** Pass `--no-conda` to rely on the active Python environment.
- **Missing baseline error:** Provide `--baseline-model <name>` to the script if automatic
  inference fails (only necessary when adding new models).
- **Hydra override tweaks:** Forward overrides after `--`, e.g.
  `./scripts/reproduce_all.sh -- --multirun.seeds=[1,2,3]`.

For additional context on the experiment methodology, consult `docs/tmlr/method_and_results.md`.

## Rebuilding the archive (maintainers)
Regenerate `artifact_anonymous.zip` from the repository root with:

```bash
python -m scripts.package_artifact --output artifact_anonymous.zip
```

The helper enforces a 100 MiB size limit and strips build metadata so repeated runs produce
identical archives.
