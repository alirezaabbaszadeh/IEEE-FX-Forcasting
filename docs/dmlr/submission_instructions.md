# DMLR Submission Instructions

Follow the steps below to prepare the IEEE FX Forecasting benchmark package for submission.

## 1. Sync repository state
- Ensure your working tree is clean: `git status` should report no pending changes.
- Pull the latest mainline updates and rebase your feature branch as needed.

## 2. Reproduce benchmark artefacts
1. Build the container image: `docker build -t ieee-fx .`.
2. Launch the evaluation workflow: `make evaluate`.
3. Generate governance reports: `python scripts/emit_governance_budget.py --output artifacts/governance.csv`.

## 3. Validate documentation
- Regenerate the manuscript with `make dmlr-pdf` and confirm `paper_outputs/ieee_fx_forecasting_manuscript.pdf` renders without LaTeX warnings.
- Verify the fairness table (`docs/dmlr/tables/fairness_metrics.csv`) and compute governance dataset (`docs/dmlr/figures/compute_governance_data.csv`) are up to date.
- Cross-check DOI and commit hash references in the manuscript before exporting the PDF.

## 4. Package submission
1. Export the archive: `tar -czf ieee-fx-submission.tar.gz docs/dmlr paper_outputs artifacts`.
2. Upload the archive together with supplementary code to the DMLR submission portal.
3. Complete the metadata form, referencing DOI `10.5281/zenodo.pending-ieee-fx` and the commit hash reported in the manuscript.

## 5. Post-submission follow-up
- Monitor reviewer questions via the portal and address reproducibility requests within 48 hours.
- Document any resubmissions in `docs/dmlr/submission_log.md` (create if absent) for audit purposes.
