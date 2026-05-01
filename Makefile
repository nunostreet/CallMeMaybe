UV := $(HOME)/.local/bin/uv
PYTHON := $(UV) run python
PYTEST := $(UV) run pytest
DEFAULT_FUNCTIONS := data/input/functions_definition.json
DEFAULT_INPUT := data/input/function_calling_tests.json
DEFAULT_OUTPUT := data/output/function_calling_results.json
SGOINFRE := $(shell [ -d /sgoinfre/$(USER) ] && echo /sgoinfre/$(USER) || echo $(HOME))
UV_ENV := UV_CACHE_DIR=$(SGOINFRE)/.cache/uv UV_PROJECT_ENVIRONMENT=$(SGOINFRE)/.venv_cmm HF_HOME=$(SGOINFRE)/.cache/huggingface

.PHONY: help install run run-large verbose debug test clean lint lint-strict

help:
	@echo "Available targets:"
	@echo "  make install      - Install project dependencies"
	@echo "  make run          - Run the project with default paths"
	@echo "  make run-large    - Run with Qwen/Qwen3-1.7B model"
	@echo "  make verbose      - Run with token-by-token decoding output"
	@echo "  make debug        - Run the project with pdb"
	@echo "  make test         - Run all tests"
	@echo "  make clean        - Remove caches and generated files"
	@echo "  make lint         - Run flake8 and required mypy checks"
	@echo "  make lint-strict  - Run flake8 and mypy --strict"

install:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	mkdir -p $(SGOINFRE)/.cache/uv
	mkdir -p $(SGOINFRE)/.venv_cmm
	mkdir -p $(SGOINFRE)/.cache/huggingface
	$(UV_ENV) $(UV) sync

run:
	$(UV_ENV) $(PYTHON) -m src \
		--functions_definition $(DEFAULT_FUNCTIONS) \
		--input $(DEFAULT_INPUT) \
		--output $(DEFAULT_OUTPUT) \
		$(ARGS)

run-large:
	$(UV_ENV) $(PYTHON) -m src \
		--functions_definition $(DEFAULT_FUNCTIONS) \
		--input $(DEFAULT_INPUT) \
		--output $(DEFAULT_OUTPUT) \
		--model Qwen/Qwen3-1.7B

verbose:
	$(UV_ENV) $(PYTHON) -m src \
		--functions_definition $(DEFAULT_FUNCTIONS) \
		--input $(DEFAULT_INPUT) \
		--output $(DEFAULT_OUTPUT) \
		--verbose

debug:
	$(UV_ENV) $(PYTHON) -m pdb -m src \
		--functions_definition $(DEFAULT_FUNCTIONS) \
		--input $(DEFAULT_INPUT) \
		--output $(DEFAULT_OUTPUT)

test:
	$(UV_ENV) $(PYTEST) $(ARGS)

clean:
	rm -rf .pytest_cache .mypy_cache
	find src tests -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	find src tests -name "*.pyc" -delete 2>/dev/null || true
	rm -f $(DEFAULT_OUTPUT)

lint:
	$(UV_ENV) $(UV) run flake8 . --exclude=.venv,llm_sdk,tests
	$(UV_ENV) $(UV) run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs \
		--exclude llm_sdk \
		--exclude .venv \
		--exclude tests

lint-strict:
	$(UV_ENV) $(UV) run flake8 . --exclude=.venv,llm_sdk,tests
	$(UV_ENV) $(UV) run mypy . --strict \
		--exclude llm_sdk \
		--exclude .venv \
		--exclude tests
