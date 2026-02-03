# Makefile for WINGS (Wavepacket Initialization on Neighboring Grid States)
# Usage: make <target>

.PHONY: help install install-dev install-gpu test test-unit test-integration test-gpu \
        test-all coverage lint format check clean build docs

PYTHON := python
PYTEST := pytest
PIP := pip

# Default target
.DEFAULT_GOAL := help

help:
	@echo "WINGS - Wavepacket Initialization on Neighboring Grid States"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Installation:"
	@echo "  install         Install package (CPU only)"
	@echo "  install-dev     Install with development dependencies"
	@echo "  install-gpu     Install with GPU support"
	@echo "  install-all     Install with all dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test            Run fast unit tests (CPU, no slow tests)"
	@echo "  test-unit       Run all unit tests"
	@echo "  test-int        Run integration tests (CPU only)"
	@echo "  test-gpu        Run GPU tests (requires CUDA)"
	@echo "  test-cusv       Run cuStateVec tests (requires cuQuantum)"
	@echo "  test-multi-gpu  Run multi-GPU tests"
	@echo "  test-slow       Run slow tests"
	@echo "  test-all        Run all tests"
	@echo "  test-parallel   Run tests in parallel"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint            Run linters (ruff, mypy)"
	@echo "  format          Format code with ruff"
	@echo "  check           Run all checks (lint + test)"
	@echo "  typecheck       Run mypy type checking"
	@echo ""
	@echo "Coverage:"
	@echo "  coverage        Run tests with coverage report"
	@echo "  coverage-full   Full coverage including GPU tests"
	@echo "  coverage-html   Generate HTML coverage report"
	@echo ""
	@echo "Build & Release:"
	@echo "  build           Build package"
	@echo "  clean           Clean build artifacts"
	@echo "  docs            Build documentation"
	@echo ""
	@echo "Development:"
	@echo "  dev             Quick development cycle (format + lint + test)"
	@echo "  watch           Watch for changes and run tests"
	@echo "  info            Show package and backend info"

# ============================================================================
# Installation
# ============================================================================

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

install-gpu:
	$(PIP) install -e ".[gpu]"

install-all:
	$(PIP) install -e ".[dev,gpu,docs]"

# ============================================================================
# Testing
# ============================================================================

# Fast unit tests (default for development)
test:
	$(PYTEST) tests/unit/ -v --tb=short \
		-m "not gpu and not custatevec and not multi_gpu and not slow" \
		--timeout=60

# All unit tests
test-unit:
	$(PYTEST) tests/unit/ -v --tb=short --timeout=120

# Integration tests (CPU only)
test-int:
	$(PYTEST) tests/integration/ -v --tb=short \
		-m "not gpu and not custatevec and not multi_gpu" \
		--timeout=300

# GPU tests (Qiskit Aer)
test-gpu:
	$(PYTEST) tests/ -v --tb=short -m "gpu" --timeout=120

# cuStateVec tests
test-cusv:
	$(PYTEST) tests/ -v --tb=short -m "custatevec" --timeout=120

# Multi-GPU tests
test-multi-gpu:
	$(PYTEST) tests/ -v --tb=short -m "multi_gpu" --timeout=300

# Slow tests
test-slow:
	$(PYTEST) tests/ -v --tb=short -m "slow" --timeout=600

# All tests
test-all:
	$(PYTEST) tests/ -v --tb=short --timeout=600

# Parallel test execution (requires pytest-xdist)
test-parallel:
	$(PYTEST) tests/unit/ -v -n auto \
		-m "not gpu and not custatevec and not multi_gpu and not slow" \
		--timeout=60

# ============================================================================
# Coverage
# ============================================================================

coverage:
	$(PYTEST) tests/unit/ \
		-m "not gpu and not custatevec and not multi_gpu and not slow" \
		--cov=src/wings \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-fail-under=60 \
		--timeout=120

coverage-full:
	$(PYTEST) tests/ \
		--cov=src/wings \
		--cov-report=term-missing \
		--cov-report=html \
		--timeout=600

coverage-html: coverage
	@echo "Opening coverage report..."
	@open htmlcov/index.html 2>/dev/null || xdg-open htmlcov/index.html 2>/dev/null || echo "Open htmlcov/index.html manually"

# ============================================================================
# Code Quality
# ============================================================================

lint:
	ruff check src/ tests/
	@echo ""
	@echo "Running mypy..."
	mypy src/wings/ --ignore-missing-imports

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/wings/ --ignore-missing-imports --show-error-codes

check: format lint test

# ============================================================================
# Build & Release
# ============================================================================

build: clean
	$(PYTHON) -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true

# ============================================================================
# Documentation
# ============================================================================

docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# ============================================================================
# Development Helpers
# ============================================================================

# Quick development cycle
dev: format lint test
	@echo ""
	@echo " All checks passed!"

# Watch tests (requires pytest-watch)
watch:
	ptw tests/unit/ -- -v --tb=short -m "not slow and not gpu"

# Quick smoke test
quick:
	$(PYTEST) tests/unit/test_config.py tests/unit/test_optimizer.py -v --tb=short -x --timeout=30

# Show package info
info:
	@echo "WINGS Package Information"
	@echo "========================="
	@$(PYTHON) -c "import wings; wings.print_backend_info()" 2>/dev/null || echo "Package not installed. Run 'make install' first."

# Verify installation
verify:
	@echo "Verifying WINGS installation..."
	@$(PYTHON) -c "from wings import GaussianOptimizer, OptimizerConfig; print(' Core imports OK')"
	@$(PYTHON) -c "from wings import optimize_gaussian_state, run_production_campaign; print(' Convenience functions OK')"
	@$(PYTHON) -c "from wings import get_backend_info; info = get_backend_info(); print(f' CPU: {info[\"cpu\"]}, GPU: {info[\"gpu_aer\"]}, cuStateVec: {info[\"custatevec\"]}')"
	@echo ""
	@echo "Installation verified successfully!"

# ============================================================================
# CI/CD Helpers
# ============================================================================

ci-test:
	$(PYTEST) tests/ \
		-m "not gpu and not custatevec and not multi_gpu" \
		--cov=src/wings \
		--cov-report=xml \
		--junitxml=junit.xml \
		--timeout=300

ci-lint:
	ruff check src/ tests/ --output-format=github
	mypy src/wings/ --ignore-missing-imports
