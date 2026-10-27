.PHONY: setup lint typecheck test datagen download-data run-api train-categoriser eval-categoriser clean

PY := uv run

setup:
	uv venv --python 3.11
	uv pip install -e ".[dev,datagen]"
	$(PY) pre-commit install

setup-ml:
	uv pip install -e ".[dev,datagen,ml]"

lint:
	$(PY) ruff check .
	$(PY) ruff format --check .

format:
	$(PY) ruff check --fix .
	$(PY) ruff format .

typecheck:
	$(PY) mypy

test:
	$(PY) pytest

datagen:
	$(PY) python -m datagen.generate --out data/processed/synthetic --personas 12 --months 24 --seed 42

download-data:
	$(PY) python scripts/download_datasets.py

run-api:
	$(PY) uvicorn fincopilot.main:app --host 127.0.0.1 --port 8765 --reload

train-categoriser:
	$(PY) python -m fincopilot.ml.categoriser.train --data data/processed --out models/categoriser

eval-categoriser:
	$(PY) python -m fincopilot.ml.categoriser.evaluate --model models/categoriser --data data/processed

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build
