# DMLR Submission Checklist

## Protocol and Data
- [ ] Data preprocessing scripts reviewed for parity with `src/data/pipelines.py`.
- [ ] Train/validation/test splits regenerated with checksum verification.
- [ ] Configuration files (`configs/models/*.yaml`, `configs/governance.yaml`) committed and tagged.

## Fairness Evaluation
- [ ] Fairness metrics recomputed and stored in `docs/dmlr/tables/fairness_metrics.csv`.
- [ ] Figure captions cross-checked against evaluation logs for consistency.
- [ ] Bias mitigation strategies documented in the manuscript Fairness section.

## Compute Governance
- [ ] Governance budget export generated via `scripts/emit_governance_budget.py`.
- [ ] Resource thresholds validated against sustainability policy thresholds.
- [ ] Incident response plan appended to governance documentation.

## Reproducibility
- [ ] Environment manifests (`environment.yml`, `requirements-dev.txt`) audited for pin coverage.
- [ ] Container build (`docker build -t ieee-fx .`) completed without errors.
- [ ] `paper_outputs/ieee_fx_forecasting_manuscript.pdf` regenerated via `make dmlr-pdf` with shell-escape enabled to capture commit hash.

## Administrative
- [ ] DOI `10.5281/zenodo.pending-ieee-fx` confirmed in submission portal.
- [ ] Repository commit hash recorded in both the manuscript and submission form.
- [ ] Submission log updated with date, contact, and version details.
