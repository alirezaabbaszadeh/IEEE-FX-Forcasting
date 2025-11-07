PYTHON ?= python
CHECK_PATHS = src scripts tests
DMLR_TEX = docs/dmlr/ieee_fx_forecasting_manuscript.tex
DMLR_BUILD_DIR = paper_outputs

.PHONY: format lint typecheck test train-smoke benchmark repro publish ci dmlr-pdf

format:
	$(PYTHON) -m black --check $(CHECK_PATHS)

lint:
	$(PYTHON) -m ruff check $(CHECK_PATHS)

typecheck:
	$(PYTHON) -m mypy $(CHECK_PATHS)

test:
	$(PYTHON) -m pytest

train-smoke:
	$(PYTHON) -m src.cli --multirun training.epochs=1 training.device=cpu data.time_steps=16 data.batch_size=32
	$(PYTHON) scripts/reproduce_all.py --populate-only

benchmark:
	$(PYTHON) scripts/benchmark.py --train-warmup 1 --inference-warmup 2 --inference-runs 5

repro: format lint typecheck test train-smoke

publish:
	mkdir -p artifacts/publish
	$(PYTHON) scripts/export_tables.py --metrics artifacts/examples/metrics.csv --output-dir artifacts/publish
	$(PYTHON) scripts/export_figures.py --metrics artifacts/examples/metrics.csv --output-dir artifacts/publish/figures
	tar -czf artifacts/publish.tar.gz -C artifacts/publish .

ci: lint typecheck test train-smoke

dmlr-pdf:
	mkdir -p $(DMLR_BUILD_DIR)
	pdflatex -shell-escape -interaction=nonstopmode -output-directory $(DMLR_BUILD_DIR) $(DMLR_TEX)
	pdflatex -shell-escape -interaction=nonstopmode -output-directory $(DMLR_BUILD_DIR) $(DMLR_TEX)
