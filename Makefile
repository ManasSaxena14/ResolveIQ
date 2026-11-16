PYTHON ?= python

.PHONY: help install install-dev lint format test coverage index eval data baseline classifier evaluate-classifier risk-features risk evaluate-risk clean

help:
	@echo "install      Install runtime dependencies and the package"
	@echo "install-dev  Install runtime, development dependencies and pre-commit hooks"
	@echo "lint         Run ruff lint and format checks"
	@echo "format       Format the code base with ruff"
	@echo "test         Run the test suite"
	@echo "coverage     Run the test suite with the coverage gate used in CI"
	@echo "index        Build the knowledge base index"
	@echo "eval         Run the RAG evaluation on the dev split"
	@echo "data         Download and prepare the ticket dataset"
	@echo "baseline     Train the TF-IDF baseline classifier"
	@echo "classifier   Fine-tune DistilBERT for every configured seed"
	@echo "evaluate-classifier  Final test-set evaluation of all classifiers"
	@echo "risk-features  Build escalation risk features from the ticket splits"
	@echo "risk         Train the escalation risk models and choose the alert threshold"
	@echo "evaluate-risk  Final test-set evaluation of the escalation risk model"
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

data:
	$(PYTHON) scripts/download_data.py
	$(PYTHON) scripts/prepare_tickets.py

baseline:
	$(PYTHON) scripts/train_baseline.py

classifier:
	$(PYTHON) scripts/train_classifier.py --all-seeds

evaluate-classifier:
	$(PYTHON) scripts/evaluate_classifier.py

risk-features:
	$(PYTHON) scripts/build_risk_features.py

risk:
	$(PYTHON) scripts/train_risk.py

evaluate-risk:
	$(PYTHON) scripts/evaluate_risk.py

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
