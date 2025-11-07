# Purged stacking appendix assets

The inference pipeline emits `stacking_weights.csv` and `stacking_fold_metrics.csv` under each
run directory inside `artifacts/`. Concatenate the per-run weight files with
`scripts/export_tables.py` to generate `stacking_appendix_table.csv` in this folder. The resulting
table mirrors the schema of the weights export and can be dropped into an appendix as Table A1.
