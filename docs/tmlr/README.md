# TMLR Manuscript Sources

## Overview
The `fx_forecasting_tmlr.tex` manuscript, auxiliary tables, and CSV inputs under this directory
mirror the anonymized submission prepared for TMLR review. Binary artifacts such as the compiled
PDF are intentionally excluded from version control; regenerate them locally when needed.

## Compiling the manuscript
1. Change into this directory:
   ```bash
   cd docs/tmlr
   ```
2. Run `pdflatex` (or `xelatex`) twice to resolve cross-references:
   ```bash
   pdflatex -interaction=nonstopmode fx_forecasting_tmlr.tex
   pdflatex -interaction=nonstopmode fx_forecasting_tmlr.tex
   ```
   The resulting `fx_forecasting_tmlr.pdf` will be ignored by Git thanks to the repository-wide
   `.gitignore` update.

## Data dependencies
The plots and tables load data from `data/*.csv` and `tables/*.tex` within this folder. Update the
CSV values first, then rebuild the LaTeX document to refresh visuals automatically via PGFPlots.
