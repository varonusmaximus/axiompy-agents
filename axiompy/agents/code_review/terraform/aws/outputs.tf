# AWS Deployment Outputs

#------------------------------------------------------------------------------
# Endpoint URLs
#------------------------------------------------------------------------------

output "webhook_url" {
  description = "Webhook URL to configure in GitHub"
  value       = var.certificate_arn != "" ? "https://${aws_lb.main.dns_name}/webhook" : "http://${aws_lb.main.dns_name}/webhook"
}

output "health_check_url" {
  description = "Health check URL"
  value       = var.certificate_arn != "" ? "https://${aws_lb.main.dns_name}/health" : "http://${aws_lb.main.dns_name}/health"
}

output "alb_dns_name" {
  description = "Application Load Balancer DNS name"
  value       = aws_lb.main.dns_name
}

#------------------------------------------------------------------------------
# Resource Identifiers
#------------------------------------------------------------------------------

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "ECS service name"
  value       = aws_ecs_service.main.name
}

output "ecr_repository_url" {
  description = "ECR repository URL for pushing Docker images"
  value       = aws_ecr_repository.code_review.repository_url
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for viewing logs"
  value       = aws_cloudwatch_log_group.ecs.name
}

output "github_token_secret_arn" {
  description = "ARN of the GitHub token secret in Secrets Manager"
  value       = aws_secretsmanager_secret.github_token.arn
  sensitive   = true
}

#------------------------------------------------------------------------------
# Configuration Summary
#------------------------------------------------------------------------------

output "gpu_enabled" {
  description = "Whether GPU acceleration is enabled"
  value       = var.enable_gpu
}

output "high_availability" {
  description = "Whether high availability (multiple NAT Gateways) is enabled"
  value       = var.high_availability
}

output "instance_type" {
  description = "Instance type used for compute"
  value       = var.enable_gpu ? var.gpu_instance_type : "Fargate (serverless)"
}

output "expected_performance" {
  description = "Expected code review performance"
  value       = var.enable_gpu ? "~20-30 seconds per file (GPU accelerated)" : "~3-5 minutes per file (CPU only)"
}

output "alb_logs_bucket" {
  description = "S3 bucket for ALB access logs (if enabled)"
  value       = var.enable_alb_logs ? aws_s3_bucket.alb_logs[0].id : null
}

#------------------------------------------------------------------------------
# Setup Instructions
#------------------------------------------------------------------------------

output "setup_instructions" {
  description = "Next steps to complete setup"
  value       = <<-EOT
    
    ✅ Infrastructure deployed!
    
    📦 STEP 1: Push Docker image to ECR:
       aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${aws_ecr_repository.code_review.repository_url}
       docker tag ghcr.io/varonusmaximus/code-review-agent:latest ${aws_ecr_repository.code_review.repository_url}:latest
       docker push ${aws_ecr_repository.code_review.repository_url}:latest
    
    🔄 STEP 2: Force ECS to pull new image:
       aws ecs update-service --cluster ${aws_ecs_cluster.main.name} --service ${aws_ecs_service.main.name} --force-new-deployment
    
    Configuration:
    - GPU Enabled: ${var.enable_gpu}
    - High Availability: ${var.high_availability}
    - Instance Type: ${var.enable_gpu ? var.gpu_instance_type : "Fargate"}
    - Model: ${var.ollama_model}
    - Expected Speed: ${var.enable_gpu ? "~20-30 seconds per file" : "~3-5 minutes per file"}
    
    Next steps after image push:
    1. Configure GitHub webhook:
       - URL: ${var.certificate_arn != "" ? "https://${aws_lb.main.dns_name}/webhook" : "http://${aws_lb.main.dns_name}/webhook"}
       - Content type: application/json
       - Events: Pull requests
    
    2. View logs:
       aws logs tail ${aws_cloudwatch_log_group.ecs.name} --follow
    
    3. Check service status:
       aws ecs describe-services --cluster ${aws_ecs_cluster.main.name} --services ${aws_ecs_service.main.name}
    
  EOT
}
