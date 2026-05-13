# Code Review Agent - AWS Deployment
#
# Deploys the Code Review Agent using:
# - ECS Fargate (or EC2 with GPU) for containers
# - Application Load Balancer for HTTPS
# - Secrets Manager for GitHub token
# - EFS for Ollama model persistence
#
# Usage:
#   cd axiompy/agents/code_review/terraform/aws
#   terraform init
#   terraform apply

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # RECOMMENDED: Uncomment and configure for production use.
  # This enables remote state storage with locking to prevent concurrent modifications.
  #
  # Prerequisites:
  #   1. Create S3 bucket: aws s3 mb s3://your-terraform-state-bucket
  #   2. Create DynamoDB table for locking:
  #      aws dynamodb create-table \
  #        --table-name terraform-locks \
  #        --attribute-definitions AttributeName=LockID,AttributeType=S \
  #        --key-schema AttributeName=LockID,KeyType=HASH \
  #        --billing-mode PAY_PER_REQUEST
  #
  # backend "s3" {
  #   bucket         = "your-terraform-state-bucket"
  #   key            = "code-review-agent/terraform.tfstate"
  #   region         = "us-west-2"
  #   encrypt        = true
  #   dynamodb_table = "terraform-locks"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "code-review-agent"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

#------------------------------------------------------------------------------
# Data Sources
#------------------------------------------------------------------------------

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

#------------------------------------------------------------------------------
# Local Variables
#------------------------------------------------------------------------------

locals {
  name_prefix = "code-review-${var.environment}"
  azs         = slice(data.aws_availability_zones.available.names, 0, 2)

  # Number of NAT gateways (1 for cost savings, 2 for HA)
  nat_gateway_count = var.high_availability ? length(local.azs) : 1

  # Use ECR image
  ecr_image_url = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/${local.name_prefix}-agent:latest"

  tags = {
    Project     = "code-review-agent"
    Environment = var.environment
  }
}
