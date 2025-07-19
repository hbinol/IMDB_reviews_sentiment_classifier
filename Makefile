# IMDB Sentiment Classifier - Makefile

.PHONY: help install clean test format lint train predict docker-up docker-down setup

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Setup and installation
setup: ## Set up the complete development environment
	./scripts/setup.sh

install: ## Install Python dependencies
	pip install -r requirements.txt

clean: ## Clean up generated files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf logs/*.log

# Data operations
check-data: ## Check and validate Kaggle dataset
	PYTHONPATH=. python scripts/check_data.py

download-data: ## Download IMDB dataset
	python -c "from src.data.data_loader import IMDBDataLoader; loader = IMDBDataLoader(); loader.download_dataset(); loader.extract_dataset()"

preprocess-data: ## Preprocess text data
	python -c "from src.data.preprocessor import TextPreprocessor; TextPreprocessor().main()"

# Training
train-quick: ## Run quick training with small dataset
	PYTHONPATH=. python -c "from src.models.train import ModelTrainer; trainer = ModelTrainer(); trainer.train_models(dataset_size=1000)"

train-full: ## Run full training pipeline
	PYTHONPATH=. python src/models/train.py

train-pipeline: ## Run Prefect training pipeline
	PYTHONPATH=. python src/models/train_pipeline.py

# Prediction
predict-sample: ## Create sample data for prediction
	PYTHONPATH=. python src/deployment/batch_predict.py --create-sample

predict-batch: ## Run batch prediction (requires model)
	PYTHONPATH=. python src/deployment/batch_predict.py --input-file data/external/sample_reviews.csv --output-file results/predictions.csv

# Testing and quality
test: ## Run tests
	PYTHONPATH=. python scripts/test_setup.py

test-full: ## Run full test suite with pytest
	pytest tests/ -v --cov=src --cov-report=html

format: ## Format code with black
	black src/ scripts/ tests/ --line-length 88

lint: ## Lint code with flake8
	flake8 src/ scripts/ tests/ --max-line-length 88 --extend-ignore E203,W503

isort: ## Sort imports
	isort src/ scripts/ tests/

quality: format lint isort ## Run all code quality tools

# Docker operations
docker-build: ## Build Docker images
	docker-compose build

docker-up: ## Start all services with Docker Compose
	docker-compose up -d

docker-down: ## Stop all services
	docker-compose down

docker-logs: ## Show Docker logs
	docker-compose logs -f

docker-clean: ## Clean Docker resources
	docker-compose down -v --remove-orphans
	docker system prune -f

# MLOps operations
mlflow-ui: ## Start MLflow UI
	mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000

prefect-server: ## Start Prefect server
	prefect server start

localstack-start: ## Start LocalStack
	localstack start

# Monitoring
monitor: ## Run model monitoring
	PYTHONPATH=. python src/monitoring/model_monitor.py

# Development
notebook: ## Start Jupyter notebook
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

dev-setup: install ## Set up development environment
	pre-commit install
	PYTHONPATH=. python scripts/test_setup.py

# Environment
env-create: ## Create conda environment
	conda create -n imdb-sentiment python=3.9 -y
	conda activate imdb-sentiment

env-export: ## Export conda environment
	conda env export > environment.yml 