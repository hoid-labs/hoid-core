# hoid local commands. Mirrors .github/workflows/ci.yaml and the Pre-PR
# Checklist in CLAUDE.md. Run `just` (no args) to list recipes.

set shell := ["bash", "-cu"]

# Default: print available recipes
default:
    @just --list

# Install uv-managed deps into a project-local venv
install:
    uv venv
    uv sync --all-extras --group dev

# Refresh Python deps from pyproject.toml without re-creating the venv
sync:
    uv sync --all-extras --group dev

# Ruff lint (matches CI: `uv run ruff check hoid tests`)
lint:
    uv run ruff check hoid tests

# Ruff lint with autofix where safe
lint-fix:
    uv run ruff check --fix hoid tests
    uv run ruff format hoid tests

# Ruff format check (no writes)
format-check:
    uv run ruff format --check hoid tests

# Bandit security scan (matches CI: `uv run bandit -r hoid`)
security:
    uv run bandit -r hoid -ll

# mypy strict type-check (CI: currently disabled, run locally per CLAUDE.md)
types:
    uv run mypy hoid

# Find unused / missing dependencies
deps-audit:
    uv run deptry hoid tests

# Unit + packaging tests, no live LLM required
test:
    uv run pytest tests/unit tests/test_packaging.py -v

# Unit + packaging tests with coverage report
test-cov:
    uv run pytest tests/unit tests/test_packaging.py --cov=hoid --cov-report=term-missing

# Full local CI pipeline: everything CI runs on a PR, sequentially
ci: lint security test
    @echo "ci: all green"

# Pre-PR Checklist from CLAUDE.md
pre-pr: lint types test
    @echo "pre-pr: all green — review CHANGELOG.md [Unreleased] entry"

# Dry-run the auto-tag script on the current branch (no push, no API calls)
release-dry:
    bash .github/scripts/auto-tag.sh "$(git rev-parse --abbrev-ref HEAD)"

# Clear tool caches (ruff, mypy, pytest, uv)
clean:
    rm -rf .ruff_cache .mypy_cache .pytest_cache .coverage
    uv cache clean
