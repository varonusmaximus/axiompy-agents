# GPU Instance Configuration
#
# EC2 instances with NVIDIA GPUs for accelerated inference.
# Only created when enable_gpu = true.

#------------------------------------------------------------------------------
# AMI for ECS-optimized GPU instances
#------------------------------------------------------------------------------

data "aws_ami" "ecs_gpu" {
  count       = var.enable_gpu ? 1 : 0
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-ecs-gpu-hvm-*-x86_64-ebs"]
  }
}

#------------------------------------------------------------------------------
# IAM Role for EC2 Instances
#------------------------------------------------------------------------------

resource "aws_iam_role" "ecs_instance" {
  count = var.enable_gpu ? 1 : 0
  name  = "${local.name_prefix}-ecs-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "ecs_instance" {
  count      = var.enable_gpu ? 1 : 0
  role       = aws_iam_role.ecs_instance[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "ecs" {
  count = var.enable_gpu ? 1 : 0
  name  = "${local.name_prefix}-ecs-instance-profile"
  role  = aws_iam_role.ecs_instance[0].name
}

#------------------------------------------------------------------------------
# Launch Template for GPU Instances
#------------------------------------------------------------------------------

resource "aws_launch_template" "gpu" {
  count         = var.enable_gpu ? 1 : 0
  name_prefix   = "${local.name_prefix}-gpu-"
  image_id      = data.aws_ami.ecs_gpu[0].id
  instance_type = var.gpu_instance_type

  iam_instance_profile {
    arn = aws_iam_instance_profile.ecs[0].arn
  }

  vpc_security_group_ids = [aws_security_group.ecs.id]

  # ECS agent configuration for GPU
  user_data = base64encode(<<-EOF
    #!/bin/bash
    echo "ECS_CLUSTER=${aws_ecs_cluster.main.name}" >> /etc/ecs/ecs.config
    echo "ECS_ENABLE_GPU_SUPPORT=true" >> /etc/ecs/ecs.config
    echo "ECS_NVIDIA_RUNTIME=nvidia" >> /etc/ecs/ecs.config
  EOF
  )

  # Enable detailed monitoring
  monitoring {
    enabled = true
  }

  tag_specifications {
    resource_type = "instance"
    tags = merge(local.tags, {
      Name = "${local.name_prefix}-gpu-instance"
    })
  }

  tags = local.tags
}

#------------------------------------------------------------------------------
# Auto Scaling Group for GPU Instances
#------------------------------------------------------------------------------

resource "aws_autoscaling_group" "gpu" {
  count               = var.enable_gpu ? 1 : 0
  name                = "${local.name_prefix}-gpu-asg"
  vpc_zone_identifier = aws_subnet.private[*].id
  min_size            = 0
  max_size            = 2
  desired_capacity    = var.desired_count

  launch_template {
    id      = aws_launch_template.gpu[0].id
    version = "$Latest"
  }

  # Protect instances during scale-in if they're running tasks
  protect_from_scale_in = true

  tag {
    key                 = "Name"
    value               = "${local.name_prefix}-gpu-instance"
    propagate_at_launch = true
  }

  tag {
    key                 = "AmazonECSManaged"
    value               = "true"
    propagate_at_launch = true
  }

  lifecycle {
    create_before_destroy = true
  }
}

#------------------------------------------------------------------------------
# ECS Capacity Provider for GPU Instances
#------------------------------------------------------------------------------

resource "aws_ecs_capacity_provider" "gpu" {
  count = var.enable_gpu ? 1 : 0
  name  = "${local.name_prefix}-gpu-provider"

  auto_scaling_group_provider {
    auto_scaling_group_arn         = aws_autoscaling_group.gpu[0].arn
    managed_termination_protection = "ENABLED"

    managed_scaling {
      maximum_scaling_step_size = 1
      minimum_scaling_step_size = 1
      status                    = "ENABLED"
      target_capacity           = 100
    }
  }

  tags = local.tags
}

