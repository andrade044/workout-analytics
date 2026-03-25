.PHONY: format lint typecheck install

lint:
	uv run ruff check src/ main.py --fix

format:
	uv run ruff format src/ main.py

typecheck:
	uv run mypy src/

install:
	uv sync --locked
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg