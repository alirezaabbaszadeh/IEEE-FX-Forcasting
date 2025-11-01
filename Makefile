.PHONY: format lint train-smoke repro publish

format:
	conda run -n ieee-fx black --check src

lint:
	conda run -n ieee-fx ruff check src

train-smoke:
	conda run -n ieee-fx python -m src.cli training.epochs=1 data.time_steps=16

repro: format lint train-smoke

publish:
	mkdir -p artifacts/publish
	conda run -n ieee-fx python scripts/export_tables.py --metrics artifacts/examples/metrics.csv --output-dir artifacts/publish
	conda run -n ieee-fx python scripts/export_figures.py --metrics artifacts/examples/metrics.csv --output-dir artifacts/publish/figures
	tar -czf artifacts/publish.tar.gz -C artifacts/publish .
