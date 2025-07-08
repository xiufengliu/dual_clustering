# Makefile for Neutrosophic Renewable Energy Forecasting Framework

.PHONY: help install install-dev test lint format clean data train evaluate experiment docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install the package"
	@echo "  install-dev  Install package with development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting (flake8, mypy)"
	@echo "  format       Format code (black, isort)"
	@echo "  clean        Clean build artifacts"
	@echo "  data         Download and prepare data"
	@echo "  train        Train the model"
	@echo "  evaluate     Evaluate the model"
	@echo "  experiment   Run full experiment"
	@echo "  docs         Generate documentation"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,notebooks,experiments]"

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v -x --disable-warnings

# Code quality
lint:
	flake8 src tests scripts experiments
	mypy src

format:
	black src tests scripts experiments notebooks
	isort src tests scripts experiments notebooks

format-check:
	black --check src tests scripts experiments notebooks
	isort --check-only src tests scripts experiments notebooks

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Data operations
data:
	python scripts/download_data.py --dataset solar --start-date 2019-01-01 --end-date 2023-10-03
	python scripts/download_data.py --dataset wind --start-date 2019-01-01 --end-date 2023-10-03

data-explore:
	python notebooks/01_data_exploration.py

# Model operations
train:
	python scripts/train_model.py --dataset solar --config base_config
	python scripts/train_model.py --dataset wind --config base_config

train-solar:
	python scripts/train_model.py --dataset solar --config solar_config

train-wind:
	python scripts/train_model.py --dataset wind --config wind_config

# Evaluation
evaluate:
	python scripts/evaluate_model.py --model results/models/solar_model_base_config.pkl --dataset solar
	python scripts/evaluate_model.py --model results/models/wind_model_base_config.pkl --dataset wind

# Experiments
experiment:
	python experiments/run_experiment.py --config base_config --dataset solar
	python experiments/run_experiment.py --config base_config --dataset wind

experiment-ablation:
	python experiments/ablation_study.py

experiment-multiple:
	python experiments/run_experiment.py --multiple --configs base_config solar_config wind_config --datasets solar wind

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "Methodology documentation available in docs/methodology.md"

# Development setup
setup-dev: install-dev
	pre-commit install

# Full pipeline
pipeline: clean install-dev data train evaluate
	@echo "Full pipeline completed successfully!"

# Quick start
quickstart:
	@echo "Setting up Neutrosophic Renewable Energy Forecasting Framework..."
	make install-dev
	make data-explore
	make train-solar
	@echo "Quick start completed! Check results/ directory for outputs."

# CI/CD targets
ci-test: format-check lint test

ci-build: clean install test

# Docker targets (if using Docker)
docker-build:
	docker build -t neutrosophic-forecasting .

docker-run:
	docker run -it --rm -v $(PWD):/workspace neutrosophic-forecasting

# Benchmarking
benchmark:
	python experiments/benchmark.py --datasets solar wind --methods all

# Model comparison
compare:
	python experiments/model_comparison.py --baseline-models naive ses arima prophet svr lstm

# Results analysis
analyze-results:
	python scripts/analyze_results.py --results-dir results/experiments

# Export results
export-results:
	python scripts/export_results.py --format latex --output results/tables/

# Profiling
profile:
	python -m cProfile -o profile_stats.prof scripts/train_model.py --dataset solar
	python -c "import pstats; pstats.Stats('profile_stats.prof').sort_stats('cumulative').print_stats(20)"

# Memory profiling
memory-profile:
	python -m memory_profiler scripts/train_model.py --dataset solar