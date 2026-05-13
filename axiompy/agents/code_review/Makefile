# Code Review Agent - Makefile
#
# Common tasks for building, testing, and deploying the code review agent.
#
# Usage:
#   make help          # Show all available targets
#   make build         # Build Docker image
#   make test          # Run tests
#   make deploy-local  # Deploy locally with Docker Compose
#   make deploy-aws    # Deploy to AWS with Terraform

.PHONY: help build test lint clean deploy-local deploy-aws destroy-aws logs shell

# Default target
.DEFAULT_GOAL := help

# Configuration
IMAGE_NAME ?= code-review-agent
IMAGE_TAG ?= latest
REGISTRY ?= ghcr.io/varonusmaximus
FULL_IMAGE := $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

# Paths (relative to this Makefile)
DOCKER_DIR := docker
TERRAFORM_AWS_DIR := terraform/aws
ROOT_DIR := ../../..

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------

help: ## Show this help message
	@echo ""
	@echo "$(BLUE)Code Review Agent - Available Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(build|test|lint|clean)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Local Deployment:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(deploy-local|logs|shell|stop)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)AWS Deployment:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(deploy-aws|destroy-aws|aws-)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Code Review:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(review-)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)CI/CD:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(push|ci-)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

#------------------------------------------------------------------------------
# Development
#------------------------------------------------------------------------------

build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	cd $(ROOT_DIR) && docker build \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		-t $(FULL_IMAGE) \
		-f axiompy/agents/code_review/$(DOCKER_DIR)/Dockerfile \
		.
	@echo "$(GREEN)✓ Image built: $(IMAGE_NAME):$(IMAGE_TAG)$(NC)"

build-no-cache: ## Build Docker image without cache
	@echo "$(BLUE)Building Docker image (no cache)...$(NC)"
	cd $(ROOT_DIR) && docker build --no-cache \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		-t $(FULL_IMAGE) \
		-f axiompy/agents/code_review/$(DOCKER_DIR)/Dockerfile \
		.
	@echo "$(GREEN)✓ Image built: $(IMAGE_NAME):$(IMAGE_TAG)$(NC)"

test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	cd $(ROOT_DIR) && python3 -m pytest tests/test_code_review.py -v --tb=short
	@echo "$(GREEN)✓ Tests passed$(NC)"

test-coverage: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	cd $(ROOT_DIR) && python3 -m pytest tests/test_code_review.py -v --cov=axiompy.agents.code_review --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)✓ Coverage report generated$(NC)"

lint: ## Run linters
	@echo "$(BLUE)Running linters...$(NC)"
	cd $(ROOT_DIR) && ruff check axiompy/agents/code_review/
	cd $(ROOT_DIR) && ruff format --check axiompy/agents/code_review/
	@echo "$(GREEN)✓ Linting passed$(NC)"

lint-fix: ## Fix linting issues
	@echo "$(BLUE)Fixing linting issues...$(NC)"
	cd $(ROOT_DIR) && ruff check --fix axiompy/agents/code_review/
	cd $(ROOT_DIR) && ruff format axiompy/agents/code_review/
	@echo "$(GREEN)✓ Linting issues fixed$(NC)"

#------------------------------------------------------------------------------
# Code Review
#------------------------------------------------------------------------------

review-file: ## Review a single file. Usage: make review-file FILE=/path/to/file.py
ifndef FILE
	@echo "$(RED)Error: FILE is required$(NC)"
	@echo "Usage: make review-file FILE=/path/to/file.py"
	@exit 1
endif
	@echo "$(BLUE)Reviewing $(FILE)...$(NC)"
	@PYTHONPATH=$(ROOT_DIR) python3 -c "\
import time; \
from axiompy.agents.code_review import CodeReviewServiceFactory; \
start = time.time(); \
service = CodeReviewServiceFactory.create_for_filesystem(rules_path='$(ROOT_DIR)/AGENTS.md'); \
result = service.review_files(['$(FILE)']); \
print(f'Time: {time.time()-start:.1f}s | Score: {result.score}/100 | Violations: {result.violation_count}')"

review-dir: ## Review all Python files in a directory. Usage: make review-dir DIR=/path/to/dir
ifndef DIR
	@echo "$(RED)Error: DIR is required$(NC)"
	@echo "Usage: make review-dir DIR=/path/to/dir"
	@exit 1
endif
	@echo "$(BLUE)Reviewing Python files in $(DIR)...$(NC)"
	@PYTHONPATH=$(ROOT_DIR) python3 -c "\
import os, time; \
from axiompy.agents.code_review import CodeReviewServiceFactory; \
files = [os.path.join(r,f) for r,d,fs in os.walk('$(DIR)') for f in fs if f.endswith('.py') and '__pycache__' not in r]; \
print(f'Found {len(files)} Python files'); \
start = time.time(); \
service = CodeReviewServiceFactory.create_for_filesystem(rules_path='$(ROOT_DIR)/AGENTS.md'); \
for f in files: \
    fstart = time.time(); \
    result = service.review_files([f]); \
    print(f'{os.path.basename(f)}: {time.time()-fstart:.1f}s | Score: {result.score} | Violations: {result.violation_count}'); \
print(f'Total: {time.time()-start:.1f}s')"

clean: ## Clean up Docker images and volumes
	@echo "$(BLUE)Cleaning up...$(NC)"
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
	docker rmi $(FULL_IMAGE) 2>/dev/null || true
	docker volume prune -f
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

#------------------------------------------------------------------------------
# Local Deployment (Docker Compose)
#------------------------------------------------------------------------------

deploy-local: ## Deploy locally with Docker Compose
	@echo "$(BLUE)Deploying locally...$(NC)"
	@if [ -z "$$GITHUB_TOKEN" ]; then \
		echo "$(RED)Error: GITHUB_TOKEN environment variable is required$(NC)"; \
		echo "Run: export GITHUB_TOKEN=ghp_your_token"; \
		exit 1; \
	fi
	cd $(DOCKER_DIR) && docker compose up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo ""
	@echo "Webhook URL: http://localhost:8080/webhook"
	@echo "Health check: http://localhost:8080/health"
	@echo ""
	@echo "View logs: make logs"

stop: ## Stop local deployment
	@echo "$(BLUE)Stopping services...$(NC)"
	cd $(DOCKER_DIR) && docker compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

stop-clean: ## Stop and remove volumes
	@echo "$(BLUE)Stopping services and removing data...$(NC)"
	cd $(DOCKER_DIR) && docker compose down -v
	@echo "$(GREEN)✓ Services stopped and data removed$(NC)"

logs: ## View logs from local deployment
	cd $(DOCKER_DIR) && docker compose logs -f

logs-review: ## View code review service logs
	cd $(DOCKER_DIR) && docker compose logs -f code-review

logs-ollama: ## View Ollama logs
	cd $(DOCKER_DIR) && docker compose logs -f ollama

shell: ## Open shell in code review container
	cd $(DOCKER_DIR) && docker compose exec code-review /bin/bash

status: ## Show status of local services
	@echo "$(BLUE)Service Status:$(NC)"
	cd $(DOCKER_DIR) && docker compose ps
	@echo ""
	@echo "$(BLUE)Health Check:$(NC)"
	@curl -s http://localhost:8080/health 2>/dev/null | python -m json.tool || echo "Service not responding"

#------------------------------------------------------------------------------
# AWS Deployment (Terraform)
#------------------------------------------------------------------------------

aws-init: ## Initialize Terraform for AWS
	@echo "$(BLUE)Initializing Terraform...$(NC)"
	cd $(TERRAFORM_AWS_DIR) && terraform init
	@echo "$(GREEN)✓ Terraform initialized$(NC)"

aws-plan: ## Plan AWS deployment
	@echo "$(BLUE)Planning AWS deployment...$(NC)"
	cd $(TERRAFORM_AWS_DIR) && terraform plan
	@echo "$(GREEN)✓ Plan complete$(NC)"

aws-validate: ## Validate Terraform configuration
	@echo "$(BLUE)Validating Terraform...$(NC)"
	cd $(TERRAFORM_AWS_DIR) && terraform validate
	@echo "$(GREEN)✓ Configuration valid$(NC)"

deploy-aws: aws-init push ## Deploy to AWS (builds, pushes, then deploys)
	@echo "$(BLUE)Deploying to AWS...$(NC)"
	cd $(TERRAFORM_AWS_DIR) && terraform apply -auto-approve
	@echo "$(GREEN)✓ Deployed to AWS$(NC)"

deploy-aws-plan: aws-init ## Deploy to AWS with plan confirmation
	@echo "$(BLUE)Deploying to AWS (with plan)...$(NC)"
	cd $(TERRAFORM_AWS_DIR) && terraform apply
	@echo "$(GREEN)✓ Deployed to AWS$(NC)"

destroy-aws: ## Destroy AWS infrastructure
	@echo "$(RED)WARNING: This will destroy all AWS resources!$(NC)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	cd $(TERRAFORM_AWS_DIR) && terraform destroy
	@echo "$(GREEN)✓ AWS resources destroyed$(NC)"

aws-output: ## Show Terraform outputs
	cd $(TERRAFORM_AWS_DIR) && terraform output

aws-logs: ## View AWS CloudWatch logs
	@echo "$(BLUE)Fetching logs from CloudWatch...$(NC)"
	@LOG_GROUP=$$(cd $(TERRAFORM_AWS_DIR) && terraform output -raw cloudwatch_log_group 2>/dev/null) && \
	aws logs tail $$LOG_GROUP --follow

#------------------------------------------------------------------------------
# CI/CD
#------------------------------------------------------------------------------

push: build ## Build and push image to registry
	@echo "$(BLUE)Pushing image to registry...$(NC)"
	docker push $(FULL_IMAGE)
	@echo "$(GREEN)✓ Image pushed: $(FULL_IMAGE)$(NC)"

ci-build: ## CI: Build image (used by GitHub Actions)
	@echo "$(BLUE)CI: Building image...$(NC)"
	cd $(ROOT_DIR) && docker build \
		-t $(FULL_IMAGE) \
		-f axiompy/agents/code_review/$(DOCKER_DIR)/Dockerfile \
		--cache-from $(FULL_IMAGE) \
		.

ci-test: ## CI: Run tests (used by GitHub Actions)
	@echo "$(BLUE)CI: Running tests...$(NC)"
	cd $(ROOT_DIR) && python3 -m pytest tests/test_code_review.py -v --junitxml=test-results.xml

ci-push: ## CI: Push image (used by GitHub Actions)
	@echo "$(BLUE)CI: Pushing image...$(NC)"
	docker push $(FULL_IMAGE)

ci-deploy: ## CI: Deploy to AWS (used by GitHub Actions)
	@echo "$(BLUE)CI: Deploying to AWS...$(NC)"
	cd $(TERRAFORM_AWS_DIR) && terraform init -input=false
	cd $(TERRAFORM_AWS_DIR) && terraform apply -auto-approve -input=false

#------------------------------------------------------------------------------
# Utilities
#------------------------------------------------------------------------------

version: ## Show version info
	@echo "Image: $(FULL_IMAGE)"
	@echo "Docker: $$(docker --version)"
	@echo "Terraform: $$(terraform --version | head -1)"

check-deps: ## Check required dependencies
	@echo "$(BLUE)Checking dependencies...$(NC)"
	@command -v docker >/dev/null 2>&1 || { echo "$(RED)✗ Docker not found$(NC)"; exit 1; }
	@command -v terraform >/dev/null 2>&1 || { echo "$(YELLOW)⚠ Terraform not found (needed for AWS deploy)$(NC)"; }
	@command -v aws >/dev/null 2>&1 || { echo "$(YELLOW)⚠ AWS CLI not found (needed for AWS deploy)$(NC)"; }
	@command -v python >/dev/null 2>&1 || { echo "$(RED)✗ Python not found$(NC)"; exit 1; }
	@echo "$(GREEN)✓ All required dependencies found$(NC)"

setup: check-deps ## Initial setup
	@echo "$(BLUE)Setting up development environment...$(NC)"
	cd $(ROOT_DIR) && pip install -e ".[dev]" 2>/dev/null || pip install -e .
	@echo "$(GREEN)✓ Setup complete$(NC)"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Set GITHUB_TOKEN: export GITHUB_TOKEN=ghp_..."
	@echo "  2. Run locally: make deploy-local"
	@echo "  3. Or deploy to AWS: make deploy-aws"

