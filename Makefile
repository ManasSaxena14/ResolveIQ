PYTHON ?= python

.PHONY: help install install-dev lint format test coverage index eval clean

help:
	@echo "install      Install runtime dependencies and the package"
	@echo "install-dev  Install runtime, development dependencies and pre-commit hooks"
	@echo "lint         Run ruff lint and format checks"
	@echo "format       Format the code base with ruff"
	@echo "test         Run the test suite"
	@echo "coverage     Run the test suite with the coverage gate used in CI"
	@echo "index        Build the knowledge base index"
	@echo "eval         Run the RAG evaluation on the dev split"
	@echo "clean        Remove caches and generated artifacts"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-deps

install-dev: install
	$(PYTHON) -m pip install -r requirements-dev.txt
	pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check . --fix

test:
	pytest

coverage:
	pytest --cov=resolveiq --cov-report=term-missing --cov-fail-under=70

index:
	$(PYTHON) scripts/build_index.py

eval:
	$(PYTHON) eval/run_rag_eval.py run --split dev

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
