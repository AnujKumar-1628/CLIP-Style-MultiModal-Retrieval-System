PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
BASH ?= bash

COMPOSE_FILE ?= docker/docker-compose.yml
COMPOSE := docker compose -f $(COMPOSE_FILE)

HOST ?= 0.0.0.0
PORT ?= 8001
SPLIT ?= test

TRAINING_CONFIG ?= configs/training.yaml
RETRIEVAL_CONFIG ?= configs/retrieval.yaml
MODEL_CONFIG ?= configs/model.yaml
DATA_CONFIG ?= configs/data.yaml
CHECKPOINT ?=

.DEFAULT_GOAL := help

.PHONY: help install install-dev test train evaluate index api \
	docker-build docker-up docker-down docker-logs docker-shell \
	docker-train docker-evaluate docker-index docker-config

help: ## Show available commands
	@echo "CLIP-style multimodal retrieval"
	@echo ""
	@echo "Local development:"
	@echo "  make install          Install Python dependencies"
	@echo "  make install-dev      Install dependencies and test tooling"
	@echo "  make test             Run the test suite"
	@echo "  make api              Start the API (HOST=... PORT=...)"
	@echo "  make train            Run training"
	@echo "  make index            Build retrieval indices (SPLIT=...)"
	@echo "  make evaluate         Evaluate retrieval (CHECKPOINT=...)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     Build the application image"
	@echo "  make docker-up        Start the API in the background"
	@echo "  make docker-down      Stop the Compose stack"
	@echo "  make docker-logs      Follow API logs"
	@echo "  make docker-shell     Open a shell in a one-off container"
	@echo "  make docker-train     Run training in Docker"
	@echo "  make docker-index     Build retrieval indices in Docker"
	@echo "  make docker-evaluate  Evaluate retrieval in Docker"
	@echo "  make docker-config    Validate and print Compose config"

install: ## Install Python dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install ## Install dependencies and test tooling
	$(PIP) install pytest

test: ## Run the test suite
	$(PYTEST) tests

api: ## Start the local API
	$(BASH) scripts/run_api.sh $(HOST) $(PORT)

train: ## Run the local training pipeline
	$(BASH) scripts/train.sh $(TRAINING_CONFIG) $(MODEL_CONFIG) $(DATA_CONFIG)

index: ## Build local retrieval indices
	$(BASH) scripts/build_index.sh $(RETRIEVAL_CONFIG) $(MODEL_CONFIG) $(DATA_CONFIG) $(SPLIT)

evaluate: ## Run local retrieval evaluation
	$(BASH) scripts/evaluate.sh $(RETRIEVAL_CONFIG) $(MODEL_CONFIG) $(DATA_CONFIG) $(CHECKPOINT)

docker-build: ## Build the application image
	$(COMPOSE) build

docker-up: ## Start the API in the background
	$(COMPOSE) up --build --detach

docker-down: ## Stop the Compose stack
	$(COMPOSE) down

docker-logs: ## Follow API logs
	$(COMPOSE) logs --follow api

docker-shell: ## Open a shell in a one-off container
	$(COMPOSE) run --rm --entrypoint sh api

docker-train: ## Run training in Docker
	$(COMPOSE) run --rm api train $(TRAINING_CONFIG) $(MODEL_CONFIG) $(DATA_CONFIG)

docker-index: ## Build retrieval indices in Docker
	$(COMPOSE) run --rm api index $(RETRIEVAL_CONFIG) $(MODEL_CONFIG) $(DATA_CONFIG) $(SPLIT)

docker-evaluate: ## Run retrieval evaluation in Docker
	$(COMPOSE) run --rm api evaluate $(RETRIEVAL_CONFIG) $(MODEL_CONFIG) $(DATA_CONFIG) $(CHECKPOINT)

docker-config: ## Validate and print the resolved Compose config
	$(COMPOSE) config
