.PHONY: help install test format lint clean

help:
	@echo "Available commands:"
	@echo "  make install   Install dependencies"
	@echo "  make test      Run test suite"
	@echo "  make format    Format code"
	@echo "  make lint      Run linter"
	@echo "  make clean     Remove generated files"

install:
	poetry install

test:
	poetry run pytest

format:
	poetry run black src tests

lint:
	poetry run ruff src tests

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	rm -rf data/cache artifacts reports models
