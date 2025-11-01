.PHONY: format lint train-smoke repro

format:
conda run -n ieee-fx black --check src

lint:
conda run -n ieee-fx ruff check src

train-smoke:
conda run -n ieee-fx python -m src.cli training.epochs=1 data.time_steps=16

repro: format lint train-smoke
