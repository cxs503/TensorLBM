# TensorLBM developer task runner.
# Common entrypoints — mirrors the CI gates so local runs match CI.
#
#   make install        Install the library + dev extras (editable)
#   make install-app    Install the FastAPI platform dependencies
#   make lint           Ruff lint (src/tests/examples/benchmarks)
#   make format         Ruff format (writes files)
#   make format-check   Ruff format --check (CI uses this)
#   make types          mypy strict on the core solver modules
#   make test           Run the fast test scope (everything except slow)
#   make test-all       Run the full suite including slow campaigns
#   make coverage       Run tests with coverage report
#   make layout         Run the repository-layout policy checker
#   make audit          Generate the import-graph architecture audit
#   make validation-matrix  Audit + judge + CI-tier run of the validation matrix (writes run ledger)
#   make models         List the model units (models/lbm/*/model.toml)
#   make models-build   Build every model unit in isolation (writes build artifacts)
#   make models-test    Run every model unit's own case against its reference
#   make models-doctor  Registry consistency gates (discovery union, generated CI matrix)
#   make model-<name>   Build + verify a single model unit (generated target)
#   make ci-local       Run the same gates CI runs, locally
#   make serve          Launch the FastAPI platform (UVICORN)
#   make clean          Remove caches and build artifacts

PYTHON ?= python
PIP    ?= pip
PYTEST ?= pytest

.PHONY: install install-app lint format format-check types test test-all coverage layout audit \
        validation-matrix ci-local serve clean \
        models models-build models-test models-doctor models-ci-matrix

install:
	$(PIP) install -e ".[dev]"

install-app:
	$(PIP) install -e ".[dev]" -r app/requirements.txt

lint:
	ruff check src tests examples benchmarks app app/tests

format:
	ruff format src tests examples benchmarks app app/tests

format-check:
	ruff format --check src tests examples benchmarks app app/tests

types:
	mypy src/tensorlbm --ignore-missing-imports

test:
	PYTHONPATH=src $(PYTEST) -q -m "not slow" --timeout=600

test-all:
	PYTHONPATH=src $(PYTEST) -q --timeout=1800

coverage:
	PYTHONPATH=src $(PYTEST) -q -m "not slow" --cov=tensorlbm --cov-report=term-missing

layout:
	$(PYTHON) tools/check_repository_layout.py

audit:
	mkdir -p artifacts/architecture
	$(PYTHON) tools/audit_import_graph.py \
		--root src/tensorlbm=tensorlbm \
		--root app/backend=app.backend \
		--root tests=tests \
		--root app/tests=app.tests \
		--json-out artifacts/architecture/import-audit.json \
		--markdown-out artifacts/architecture/import-audit.md

validation-matrix:
	mkdir -p artifacts/validation_matrix/runs
	PYTHONPATH=src $(PYTHON) -m benchmarks.validation_matrix audit
	PYTHONPATH=src $(PYTHON) -m benchmarks.validation_matrix check
	PYTHONPATH=src $(PYTHON) -m benchmarks.validation_matrix run --tier ci
	PYTHONPATH=src $(PYTHON) -m benchmarks.validation_matrix status --write
	PYTHONPATH=src $(PYTHON) benchmarks/validation_matrix/make_benchmark_table.py --check
	PYTHONPATH=src $(PYTHON) benchmarks/validation_matrix/runledger.py verify

# --- model units -------------------------------------------------------------
# Ported from TCLB: the per-model build targets are *generated* from the model
# directories (TCLB regenerates ``makefile.main`` when ``models/*`` changes), so
# adding a model never requires editing this Makefile.
MODELS_MK := artifacts/model_units/Makefile.models
MODEL_MANIFESTS := $(wildcard models/lbm/*/model.toml)

-include $(MODELS_MK)

$(MODELS_MK): $(MODEL_MANIFESTS)
	@mkdir -p $(dir $@)
	PYTHONPATH=src $(PYTHON) -m tensorlbm.model_units makefile --out $@

models:
	PYTHONPATH=src $(PYTHON) -m tensorlbm.model_units list

models-build:
	PYTHONPATH=src $(PYTHON) -m tensorlbm.model_units build

models-test:
	PYTHONPATH=src $(PYTHON) -m tensorlbm.model_units test

models-ci-matrix:
	PYTHONPATH=src $(PYTHON) -m tensorlbm.model_units ci-matrix --out artifacts/model_units/ci_matrix.json

models-doctor:
	PYTHONPATH=src $(PYTHON) -m tensorlbm.model_units doctor

ci-local: lint format-check types test layout models-doctor

serve:
	PYTHONPATH=src uvicorn app.backend.main:app --reload --host 0.0.0.0 --port 8000

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	rm -rf artifacts/architecture artifacts/ci coverage.xml htmlcov
