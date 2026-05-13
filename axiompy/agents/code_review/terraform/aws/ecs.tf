# ECS Resources
#
# ECS cluster, task definition, service, and CloudWatch logging.
# Supports both Fargate (CPU) and EC2 (GPU) launch types.

#------------------------------------------------------------------------------
# Local Variables for Container Configuration
#------------------------------------------------------------------------------

locals {
  # Base secrets (always include GitHub token)
  base_secrets = [{
    name      = "GITHUB_TOKEN"
    valueFrom = aws_secretsmanager_secret.github_token.arn
  }]

  # Webhook secret (only if configured)
  webhook_secret = var.webhook_secret != "" ? [{
    name      = "WEBHOOK_SECRET"
    valueFrom = aws_secretsmanager_secret.webhook_secret[0].arn
  }] : []

  # Combined secrets list
  container_secrets = concat(local.base_secrets, local.webhook_secret)
}

#------------------------------------------------------------------------------
# CloudWatch Logs
#------------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${local.name_prefix}"
  retention_in_days = 30

  tags = local.tags
}

#------------------------------------------------------------------------------
# ECS Cluster
#------------------------------------------------------------------------------

resource "aws_ecs_cluster" "main" {
  name = "${local.name_prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = local.tags
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name = aws_ecs_cluster.main.name

  capacity_providers = var.enable_gpu ? [aws_ecs_capacity_provider.gpu[0].name] : ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = var.enable_gpu ? aws_ecs_capacity_provider.gpu[0].name : "FARGATE"
    weight            = 1
    base              = 1
  }
}

#------------------------------------------------------------------------------
# ECS Task Definition
#------------------------------------------------------------------------------

resource "aws_ecs_task_definition" "main" {
  family                   = local.name_prefix
  network_mode             = "awsvpc"
  requires_compatibilities = var.enable_gpu ? ["EC2"] : ["FARGATE"]
  cpu                      = var.enable_gpu ? 4096 : 4096  # 4 vCPU for faster inference
  memory                   = var.enable_gpu ? 15360 : 16384 # 16GB for model + inference
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  volume {
    name = "ollama-data"

    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.ollama.id
      transit_encryption = "ENABLED"
      authorization_config {
        access_point_id = aws_efs_access_point.ollama.id
        iam             = "ENABLED"
      }
    }
  }

  # GPU container definitions (with NVIDIA runtime)
  container_definitions = var.enable_gpu ? jsonencode([
    {
      name      = "ollama"
      image     = "ollama/ollama:latest"
      essential = true

      # Request GPU resources
      resourceRequirements = [{
        type  = "GPU"
        value = "1"
      }]

      portMappings = [{
        containerPort = 11434
        protocol      = "tcp"
      }]

      mountPoints = [{
        sourceVolume  = "ollama-data"
        containerPath = "/root/.ollama"
        readOnly      = false
      }]

      # Override entrypoint to run shell commands for model pull
      entryPoint = ["/bin/sh", "-c"]
      command    = ["ollama serve & sleep 10 && ollama pull ${var.ollama_model} && wait"]

      healthCheck = {
        command     = ["CMD-SHELL", "ollama list || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 10
        startPeriod = 300
      }

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ollama-gpu"
        }
      }
    },
    {
      name      = "code-review"
      image     = local.ecr_image_url
      essential = true

      dependsOn = [{
        containerName = "ollama"
        condition     = "HEALTHY"
      }]

      portMappings = [{
        containerPort = 8080
        protocol      = "tcp"
      }]

      environment = [
        { name = "OLLAMA_HOST", value = "http://localhost:11434" },
        { name = "OLLAMA_MODEL", value = var.ollama_model },
        { name = "RULES_REPO", value = var.rules_repo },
        { name = "RULES_FILE", value = var.rules_file },
        { name = "LOG_LEVEL", value = var.log_level },
        { name = "GPU_ENABLED", value = "true" },
      ]

      secrets = local.container_secrets

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 30
      }

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "code-review"
        }
      }
    }
    ]) : jsonencode([
    # CPU-only (Fargate) container definitions
    {
      name      = "ollama"
      image     = "ollama/ollama:latest"
      essential = true

      portMappings = [{
        containerPort = 11434
        protocol      = "tcp"
      }]

      mountPoints = [{
        sourceVolume  = "ollama-data"
        containerPath = "/root/.ollama"
        readOnly      = false
      }]

      # Override entrypoint to run shell commands
      entryPoint = ["/bin/sh", "-c"]
      command    = ["ollama serve & sleep 10 && ollama pull ${var.ollama_model} && wait"]

      healthCheck = {
        command     = ["CMD-SHELL", "ollama list || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 10
        startPeriod = 300
      }

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ollama"
        }
      }
    },
    {
      name      = "code-review"
      image     = local.ecr_image_url
      essential = true

      dependsOn = [{
        containerName = "ollama"
        condition     = "HEALTHY"
      }]

      portMappings = [{
        containerPort = 8080
        protocol      = "tcp"
      }]

      environment = [
        { name = "OLLAMA_HOST", value = "http://localhost:11434" },
        { name = "OLLAMA_MODEL", value = var.ollama_model },
        { name = "RULES_REPO", value = var.rules_repo },
        { name = "RULES_FILE", value = var.rules_file },
        { name = "LOG_LEVEL", value = var.log_level },
      ]

      secrets = local.container_secrets

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 30
      }

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "code-review"
        }
      }
    }
  ])

  tags = local.tags
}

#------------------------------------------------------------------------------
# ECS Service
#------------------------------------------------------------------------------

resource "aws_ecs_service" "main" {
  name            = local.name_prefix
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.main.arn
  desired_count   = var.desired_count

  # Use EC2 for GPU, Fargate for CPU-only
  launch_type = var.enable_gpu ? null : "FARGATE"

  # Capacity provider strategy for GPU instances
  dynamic "capacity_provider_strategy" {
    for_each = var.enable_gpu ? [1] : []
    content {
      capacity_provider = aws_ecs_capacity_provider.gpu[0].name
      weight            = 1
      base              = 1
    }
  }

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.main.arn
    container_name   = "code-review"
    container_port   = 8080
  }

  # Enable deployment circuit breaker for faster rollback on failures
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  # Ensure listeners and EFS are ready before creating service
  depends_on = [
    aws_lb_listener.http_webhook,
    aws_lb_listener.https,
    aws_efs_mount_target.ollama
  ]

  tags = local.tags
}

