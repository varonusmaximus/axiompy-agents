# Storage Resources
#
# ECR repository for Docker images, Secrets Manager for GitHub token,
# and EFS for persistent Ollama model storage.

#------------------------------------------------------------------------------
# ECR Repository for Code Review Agent
#------------------------------------------------------------------------------

resource "aws_ecr_repository" "code_review" {
  name                 = "${local.name_prefix}-agent"
  image_tag_mutability = "MUTABLE"
  force_delete         = true # Allow deletion even with images

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = local.tags
}

# Allow ECS to pull from ECR
resource "aws_ecr_repository_policy" "code_review" {
  repository = aws_ecr_repository.code_review.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowECSPull"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability"
        ]
      }
    ]
  })
}

#------------------------------------------------------------------------------
# Secrets Manager
#------------------------------------------------------------------------------

resource "aws_secretsmanager_secret" "github_token" {
  name                    = "${local.name_prefix}-github-token"
  description             = "GitHub token for Code Review Agent"
  recovery_window_in_days = 0 # Allow immediate deletion for dev

  tags = local.tags
}

resource "aws_secretsmanager_secret_version" "github_token" {
  secret_id     = aws_secretsmanager_secret.github_token.id
  secret_string = var.github_token
}

# Webhook secret for verifying GitHub signatures (optional but recommended)
resource "aws_secretsmanager_secret" "webhook_secret" {
  count                   = var.webhook_secret != "" ? 1 : 0
  name                    = "${local.name_prefix}-webhook-secret"
  description             = "Webhook secret for verifying GitHub signatures"
  recovery_window_in_days = 0

  tags = local.tags
}

resource "aws_secretsmanager_secret_version" "webhook_secret" {
  count         = var.webhook_secret != "" ? 1 : 0
  secret_id     = aws_secretsmanager_secret.webhook_secret[0].id
  secret_string = var.webhook_secret
}

#------------------------------------------------------------------------------
# EFS for Ollama Models
#------------------------------------------------------------------------------

resource "aws_efs_file_system" "ollama" {
  creation_token = "${local.name_prefix}-ollama-efs"
  encrypted      = true

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }

  tags = {
    Name = "${local.name_prefix}-ollama-efs"
  }
}

resource "aws_efs_mount_target" "ollama" {
  count           = length(local.azs)
  file_system_id  = aws_efs_file_system.ollama.id
  subnet_id       = aws_subnet.private[count.index].id
  security_groups = [aws_security_group.efs.id]
}

resource "aws_efs_access_point" "ollama" {
  file_system_id = aws_efs_file_system.ollama.id

  posix_user {
    uid = 1000
    gid = 1000
  }

  root_directory {
    path = "/ollama"
    creation_info {
      owner_uid   = 1000
      owner_gid   = 1000
      permissions = "755"
    }
  }

  tags = {
    Name = "${local.name_prefix}-ollama-ap"
  }
}

