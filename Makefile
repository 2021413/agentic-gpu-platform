# ---------------------------------------------------------------------------
# Agentic GPU Platform - developer entry points.
# `make` with no target prints the list below.
# ---------------------------------------------------------------------------

SHELL := /bin/bash
.DEFAULT_GOAL := help

VENV    ?= .venv
BIN     := $(VENV)/bin
PYTHON  ?= python3
COMPOSE ?= docker compose

# Unit tests are everything that needs neither Docker nor a GPU.
UNIT_SELECTOR := not integration and not e2e

# Local dev server (same fallback logic as docker/entrypoint-api.sh).
API_APP  ?= bootstrap.app:create_app
API_HOST ?= 127.0.0.1
API_PORT ?= 8000

##@ Setup

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage: make \033[36m<target>\033[0m\n"} \
	/^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 } \
	/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
	@echo

install: ## Create .venv and install the project with its dev dependencies
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev]"
	$(BIN)/pre-commit install

hooks: ## (Re)install the pre-commit hooks
	$(BIN)/pre-commit install

hooks-run: ## Run every pre-commit hook on the whole tree
	$(BIN)/pre-commit run --all-files

##@ Quality

lint: ## Ruff lint + format check
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

format: ## Apply Ruff formatting and safe autofixes
	$(BIN)/ruff format src tests
	$(BIN)/ruff check --fix src tests

typecheck: ## mypy (strict, configured in pyproject.toml)
	$(BIN)/mypy

check: lint typecheck test ## Everything CI runs without Docker

##@ Tests

test: ## Unit tests (no Docker, no GPU)
	$(BIN)/pytest -m "$(UNIT_SELECTOR)"

test-integration: ## Integration tests (needs PostgreSQL + Redis, i.e. a Docker daemon)
	$(BIN)/pytest -m integration

test-e2e: ## End-to-end scenario against the fake worker
	$(BIN)/pytest -m e2e

test-all: ## Every test, all markers included
	$(BIN)/pytest

coverage: ## Unit tests with a coverage report
	$(BIN)/pytest -m "$(UNIT_SELECTOR)" --cov=src --cov-report=term-missing

##@ Run

dev: ## Postgres + Redis in Docker, API on the host with autoreload
	$(COMPOSE) up -d postgres redis
	$(BIN)/uvicorn $(API_APP) --factory --reload --host $(API_HOST) --port $(API_PORT)

migrate: ## Apply database migrations (alembic upgrade head)
	$(BIN)/alembic upgrade head

migration: ## Autogenerate a migration: make migration m="add runs table"
	$(BIN)/alembic revision --autogenerate -m "$(m)"

##@ Docker

docker-build: ## Build the API image (the GPU image is built by docker-up-gpu)
	$(COMPOSE) build api

docker-up: ## Start the local stack: api, postgres, redis, fake worker (no GPU)
	$(COMPOSE) up -d --build
	$(COMPOSE) ps

docker-up-gpu: ## Start the stack plus a real vLLM GPU worker (NVIDIA toolkit required)
	$(COMPOSE) --profile gpu up -d --build

docker-down: ## Stop the stack (named volumes are kept)
	$(COMPOSE) --profile gpu down --remove-orphans

docker-reset: ## Stop the stack AND delete its volumes (Postgres data, model cache)
	$(COMPOSE) --profile gpu down --remove-orphans --volumes

docker-logs: ## Follow the logs of the whole stack
	$(COMPOSE) logs -f --tail=100

##@ Housekeeping

clean: ## Remove caches, build artifacts and compiled files
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -path ./$(VENV) -prune -o -type d -name __pycache__ -print0 | xargs -0 rm -rf
	find . -path ./$(VENV) -prune -o -type f -name '*.py[co]' -print0 | xargs -0 rm -f

.PHONY: help install hooks hooks-run lint format typecheck check test test-integration \
        test-e2e test-all coverage dev migrate migration docker-build docker-up \
        docker-up-gpu docker-down docker-reset docker-logs clean
