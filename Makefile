.PHONY: dev test test-fast test-integration test-cov lint format type-check check fix clean \
       build build-sdk build-mcp \
       db-start db-stop db-status db-setup db-reset serve-stdio serve-http \
       poc-hebbian poc-spread poc-token-reseed help

# Load .env for all uv run commands (API keys, DATABASE_URL, etc.)
UV_RUN := uv run --env-file .env

# -- Setup --

dev:
	uv sync --dev
	@echo "Dependencies installed."

# -- Testing --

test:
	$(UV_RUN) pytest $(if $(FILE),$(FILE),)

test-fast:
	$(UV_RUN) pytest $(if $(FILE),$(FILE),tests/unit/) -x --tb=short -m "not slow"

test-cov:
	$(UV_RUN) pytest --cov --cov-report=html --cov-report=term

test-integration:
	$(UV_RUN) pytest tests/integration/ -m slow --tb=short $(if $(FILE),$(FILE),)

# -- Code Quality --

lint:
	$(UV_RUN) ruff check $(if $(FILE),$(FILE),packages/memory/src/recollect/ packages/memory-mcp/src/recollect_mcp/ tests/)

format:
	$(UV_RUN) ruff format $(if $(FILE),$(FILE),packages/memory/src/recollect/ packages/memory-mcp/src/recollect_mcp/ tests/)

type-check:
	$(UV_RUN) mypy $(if $(FILE),$(FILE),packages/memory/src/recollect/ packages/memory-mcp/src/recollect_mcp/) --strict

check: lint type-check test

fix:
	$(UV_RUN) ruff check --fix packages/memory/src/recollect/ packages/memory-mcp/src/recollect_mcp/ tests/
	$(UV_RUN) ruff format packages/memory/src/recollect/ packages/memory-mcp/src/recollect_mcp/ tests/

# -- Database --

PG_BIN := /opt/homebrew/opt/postgresql@17/bin

db-start:
	brew services start postgresql@17

db-stop:
	brew services stop postgresql@17

db-status:
	@$(PG_BIN)/pg_isready -q && echo "PostgreSQL running" || echo "PostgreSQL stopped"

db-setup:
	$(PG_BIN)/createdb memory_v3 2>/dev/null || true
	$(PG_BIN)/psql -d memory_v3 -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null
	@echo "Database memory_v3 ready with pgvector."

db-reset:
	$(PG_BIN)/dropdb memory_v3 2>/dev/null || true
	$(PG_BIN)/createdb memory_v3
	$(PG_BIN)/psql -d memory_v3 -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null
	@echo "Database memory_v3 recreated fresh with pgvector."

# -- POC Experiments --
# Usage: make poc-spread ARGS="--spread-decay 0.8 --iter-max-rounds 5"

poc-hebbian:
	$(UV_RUN) python experiments/hebbian_poc/benchmark.py $(ARGS)

poc-spread:
	$(UV_RUN) python experiments/iterative_spread_poc/benchmark.py $(ARGS)

poc-token-reseed:
	$(UV_RUN) python experiments/token_reseed_poc/benchmark.py $(ARGS)

# -- Server --

serve-stdio:
	$(UV_RUN) python -m recollect_mcp.server

serve-http:
	$(UV_RUN) python -m recollect_mcp.server --transport streamable-http

# -- Build --

build:
	rm -rf dist/
	uv build --all

build-sdk:
	uv build --package recollect

build-mcp:
	uv build --package recollect-mcp

# -- Maintenance --

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	rm -rf dist/ build/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/

# -- Help --

help:
	@echo "Memory SDK v3"
	@echo ""
	@echo "Setup:"
	@echo "  make dev            Install dependencies"
	@echo ""
	@echo "Testing:              (FILE=path for single file)"
	@echo "  make test           Run all tests"
	@echo "  make test-fast      Unit tests, stop on first failure"
	@echo "  make test-cov       Tests with coverage report"
	@echo "  make test-integration Run integration tests (needs PostgreSQL)"
	@echo ""
	@echo "Quality:              (FILE=path for single file)"
	@echo "  make lint           Ruff lint check"
	@echo "  make format         Ruff format"
	@echo "  make type-check     Mypy strict"
	@echo "  make check          All three + tests"
	@echo "  make fix            Auto-fix lint + format"
	@echo ""
	@echo "Database:"
	@echo "  make db-start       Start PostgreSQL 17"
	@echo "  make db-stop        Stop PostgreSQL 17"
	@echo "  make db-status      Check if running"
	@echo "  make db-setup       Create memory_v3 db + pgvector"
	@echo "  make db-reset       Drop and recreate memory_v3"
	@echo ""
	@echo "POC Experiments:      (ARGS= for extra flags)"
	@echo "  make poc-hebbian          Hebbian recall tokens benchmark"
	@echo "  make poc-spread           Iterative spreading activation benchmark"
	@echo "  make poc-token-reseed     Iterative token re-seeding benchmark"
	@echo "    ARGS='--spread-decay 0.8 --iter-max-rounds 5'"
	@echo ""
	@echo "Server:"
	@echo "  make serve-stdio    MCP server (stdio transport)"
	@echo "  make serve-http     MCP server (streamable-http)"
	@echo ""
	@echo "Build:"
	@echo "  make build          Build all packages"
	@echo "  make build-sdk      Build recollect only"
	@echo "  make build-mcp      Build recollect-mcp only"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          Remove build artifacts"
