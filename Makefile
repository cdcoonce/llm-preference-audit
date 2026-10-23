.PHONY: setup lint format test leak-check ci data data-kaggle split profile

setup:
	uv sync --group dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

leak-check:
	uv run python scripts/check_data_leak.py

ci:
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) leak-check

data:
	uv run python scripts/download_data.py

data-kaggle:
	uv run python scripts/download_kaggle.py
