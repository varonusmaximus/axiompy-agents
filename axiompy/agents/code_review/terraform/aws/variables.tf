# AWS Deployment Variables

#------------------------------------------------------------------------------
# Required Variables
#------------------------------------------------------------------------------

variable "github_token" {
  description = "GitHub token for posting reviews (stored in Secrets Manager)"
  type        = string
  sensitive   = true
}

variable "webhook_secret" {
  description = "GitHub webhook secret for verifying request signatures (recommended for production)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "rules_repo" {
  description = "GitHub repository containing AGENTS.md (format: owner/repo)"
  type        = string

  validation {
    condition     = can(regex("^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$", var.rules_repo))
    error_message = "rules_repo must be in format 'owner/repo'"
  }
}

#------------------------------------------------------------------------------
# AWS Configuration
#------------------------------------------------------------------------------

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"

  validation {
    condition     = contains(["production", "staging", "development"], var.environment)
    error_message = "environment must be one of: production, staging, development"
  }
}

#------------------------------------------------------------------------------
# Networking
#------------------------------------------------------------------------------

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "high_availability" {
  description = "Enable high availability with multiple NAT Gateways. Set to false for dev/staging to save ~$32/month per NAT Gateway."
  type        = bool
  default     = true
}

variable "certificate_arn" {
  description = "ACM certificate ARN for HTTPS (leave empty for HTTP only)"
  type        = string
  default     = ""
}

#------------------------------------------------------------------------------
# Container Configuration
#------------------------------------------------------------------------------

variable "code_review_image" {
  description = "Docker image for code review service"
  type        = string
  default     = "ghcr.io/varonusmaximus/code-review-agent:latest"
}

variable "ollama_model" {
  description = "Ollama model to use (any model from https://ollama.com/library). Examples: qwen2.5-coder:1.5b, codellama, mistral, llama3.2:1b"
  type        = string
  default     = "qwen2.5-coder:1.5b"
}

variable "rules_file" {
  description = "Path to rules file in repository"
  type        = string
  default     = "AGENTS.md"
}

variable "log_level" {
  description = "Logging level"
  type        = string
  default     = "INFO"

  validation {
    condition     = contains(["DEBUG", "INFO", "WARNING", "ERROR"], var.log_level)
    error_message = "log_level must be one of: DEBUG, INFO, WARNING, ERROR"
  }
}

#------------------------------------------------------------------------------
# Scaling & Performance
#------------------------------------------------------------------------------

variable "desired_count" {
  description = "Number of ECS tasks to run"
  type        = number
  default     = 1
}

variable "enable_gpu" {
  description = "Enable GPU instance for 10-20x faster inference. Uses EC2 instead of Fargate."
  type        = bool
  default     = true # GPU enabled by default for production performance
}

variable "gpu_instance_type" {
  description = "EC2 instance type for GPU (g5.xlarge = A10G GPU, cost-effective)"
  type        = string
  default     = "g5.xlarge"

  validation {
    condition     = can(regex("^(g4dn|g5|p3|p4d)\\.", var.gpu_instance_type))
    error_message = "gpu_instance_type must be a GPU instance (g4dn, g5, p3, or p4d family)"
  }
}

#------------------------------------------------------------------------------
# Observability
#------------------------------------------------------------------------------

variable "enable_alb_logs" {
  description = "Enable ALB access logs to S3 for debugging webhook issues"
  type        = bool
  default     = false
}
