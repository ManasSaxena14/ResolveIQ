PYTHON ?= python

.PHONY: help install install-dev lint format test coverage index eval data baseline classifier evaluate-classifier risk-features risk evaluate-risk register api ui drift-reference drift benchmark evaluate-groups web-install web-dev web-build web-test space deploy-space oracle-push clean

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
	@echo "register     Register the production models in the MLflow model registry"
	@echo "api          Start the FastAPI service on http://127.0.0.1:8000"
	@echo "ui           Start the Gradio web UI on http://127.0.0.1:7860"
	@echo "drift-reference  Build the drift reference profile from the validation split"
	@echo "drift        Compare recent requests with the reference and write a drift report"
	@echo "benchmark    Measure response time of a running API (start it with make api)"
	@echo "evaluate-groups  Compare classifier and risk results across ticket types"
	@echo "web-install  Install the web app dependencies (needs Node.js 20 or later)"
	@echo "web-dev      Start the web app with hot reload on http://127.0.0.1:5173 (run make api too)"
	@echo "web-build    Build the web app; make api then serves it on http://127.0.0.1:8000"
	@echo "web-test     Lint, type check and test the web app"
	@echo "space        Run the public demo launcher locally on http://127.0.0.1:7860"
	@echo "deploy-space Deploy to Hugging Face Spaces (needs a paid HF plan): SPACE=... PAID=1"
	@echo "oracle-push  Copy the models and web app to the Oracle Cloud server: HOST=ubuntu@IP"
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

register:
	$(PYTHON) scripts/register_models.py

api:
	uvicorn resolveiq.api.main:app --host 127.0.0.1 --port 8000

ui:
	$(PYTHON) app.py

drift-reference:
	$(PYTHON) scripts/drift_report.py --build-reference

drift:
	$(PYTHON) scripts/drift_report.py

benchmark:
	$(PYTHON) scripts/benchmark_api.py --requests 50 --output results/system/api_benchmark_local.json

evaluate-groups:
	$(PYTHON) scripts/evaluate_groups.py

web-install:
	cd frontend && npm ci

web-dev:
	cd frontend && npm run dev

web-build:
	cd frontend && npm run build

web-test:
	cd frontend && npm run lint && npm run typecheck && npm test

space:
	$(PYTHON) space.py

deploy-space:
	$(PYTHON) scripts/deploy_space.py --space $(SPACE) $(if $(PAID),--paid-plan,--dry-run) $(if $(CLASSIFIER),--set-variables $(CLASSIFIER) $(RISK),)

oracle-push: web-build
	bash deploy/oracle/push.sh $(HOST)

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
