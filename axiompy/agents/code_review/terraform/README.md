# Code Review Agent - Terraform Deployment

Infrastructure-as-code for deploying the Code Review Agent to AWS.

## Overview

Deploy the AI-powered code review agent with Ollama to AWS using Terraform. The configuration creates a production-ready setup with ECS (Fargate or EC2 with GPU), Application Load Balancer, and managed secrets.

### Architecture

```
                                    GitHub
                                      │
                                      │ Webhook: "PR #42 opened"
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                    AWS                                       │
│                                                                              │
│   Public Subnets                              Private Subnets                │
│  ┌─────────────────────┐                    ┌─────────────────────────────┐ │
│  │                     │                    │                             │ │
│  │  ┌───────────────┐  │                    │  ┌───────────────────────┐  │ │
│  │  │      ALB      │  │   Forwards to      │  │   ECS Fargate / EC2   │  │ │
│  │  │  (port 443)   │──┼───────────────────▶│  │                       │  │ │
│  │  └───────────────┘  │   port 8080        │  │  ┌─────────────────┐  │  │ │
│  │                     │                    │  │  │  Code Review    │  │  │ │
│  │  ┌───────────────┐  │                    │  │  │  Service :8080  │  │  │ │
│  │  │ NAT Gateway   │  │                    │  │  └────────┬────────┘  │  │ │
│  │  │ (outbound)    │◀─┼────────────────────│  │           │           │  │ │
│  │  └───────────────┘  │                    │  │  ┌────────▼────────┐  │  │ │
│  │                     │                    │  │  │     Ollama      │  │  │ │
│  └─────────────────────┘                    │  │  │  (qwen2.5-coder)│  │  │ │
│                                             │  │  │    :11434       │  │  │ │
│                                             │  │  └─────────────────┘  │  │ │
│                                             │  │                       │  │ │
│                                             │  └───────────────────────┘  │ │
│                                             │              │              │ │
│                                             └──────────────┼──────────────┘ │
│                                                            │                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────▼──────────────┐ │
│  │ Secrets Manager │  │   CloudWatch    │  │           EFS              │ │
│  │ (GITHUB_TOKEN)  │  │    (Logs)       │  │   (Ollama model storage)   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Traffic Flow

1. **GitHub → ALB**: Webhook calls public ALB endpoint
2. **ALB → ECS**: ALB forwards to containers in private subnet
3. **ECS → GitHub**: Service calls GitHub API via NAT Gateway
4. **ECS → Ollama**: Code review service calls Ollama (same task)

---

## Quick Start

```bash
cd axiompy/agents/code_review/terraform/aws

# 1. Initialize Terraform
terraform init

# 2. Configure variables
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars:
#   github_token = "ghp_your_token"
#   rules_repo   = "varonusmaximus/axiompy"

# 3. Preview changes
terraform plan

# 4. Deploy
terraform apply
```

After deployment, Terraform outputs the webhook URL:

```
webhook_url = "http://code-review-prod-alb-123456.us-west-2.elb.amazonaws.com/webhook"
```

Configure this URL in your GitHub repository webhooks.

---

## File Structure

```
terraform/
├── README.md                 # This file
└── aws/
    ├── main.tf               # Provider, backend, data sources, locals
    ├── vpc.tf                # VPC, subnets, NAT gateways, routes
    ├── security.tf           # Security groups (ALB, ECS, EFS)
    ├── storage.tf            # ECR, Secrets Manager, EFS
    ├── iam.tf                # IAM roles and policies
    ├── ecs.tf                # ECS cluster, task definition, service
    ├── gpu.tf                # GPU EC2 instances (conditional)
    ├── alb.tf                # ALB, listeners, target groups
    ├── variables.tf          # Input variables
    ├── outputs.tf            # Output values
    └── terraform.tfvars.example
```

---

## Configuration

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `github_token` | GitHub token for posting reviews | `ghp_xxxx` |
| `rules_repo` | Repository containing AGENTS.md | `varonusmaximus/axiompy` |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `aws_region` | `us-west-2` | AWS region |
| `environment` | `production` | Environment name (production/staging/development) |
| `ollama_model` | `qwen2.5-coder:1.5b` | Ollama model to use |
| `desired_count` | `1` | Number of ECS tasks |
| `enable_gpu` | `true` | Use GPU for 10-20x faster inference |
| `gpu_instance_type` | `g5.xlarge` | GPU instance type |
| `high_availability` | `true` | Multiple NAT Gateways for HA |
| `certificate_arn` | `""` | ACM certificate for HTTPS (optional) |
| `enable_alb_logs` | `false` | Enable ALB access logs to S3 |
| `vpc_cidr` | `10.0.0.0/16` | VPC CIDR block |

### Example terraform.tfvars

```hcl
# Required
github_token = "ghp_your_token_here"
rules_repo   = "varonusmaximus/axiompy"

# Performance
enable_gpu        = true
gpu_instance_type = "g5.xlarge"
ollama_model      = "qwen2.5-coder:1.5b"

# Cost optimization (for dev/staging)
high_availability = false  # Single NAT Gateway saves ~$32/mo

# Optional
aws_region    = "us-west-2"
environment   = "production"
desired_count = 1

# For HTTPS (recommended for production)
# certificate_arn = "arn:aws:acm:us-west-2:123456789:certificate/abc-123"
```

---

## Resources Created

| Resource | Purpose | Cost Factor |
|----------|---------|-------------|
| VPC + Subnets | Network isolation | Free |
| Internet Gateway | Inbound traffic | Free |
| NAT Gateway (x1-2) | Outbound from private subnet | ~$32/mo each |
| ALB | Public webhook endpoint | ~$16/mo |
| ECS Cluster | Container orchestration | Free |
| ECS Service (Fargate) | Runs containers (CPU) | ~$73/mo (2 vCPU, 8GB) |
| EC2 GPU Instance | Runs containers (GPU) | ~$750/mo (g5.xlarge) |
| EFS | Ollama model storage | ~$0.30/GB |
| Secrets Manager | GitHub token storage | ~$0.40/mo |
| CloudWatch Logs | Logging | ~$0.50/GB |
| S3 (optional) | ALB access logs | ~$0.023/GB |

### Estimated Monthly Cost

| Configuration | Cost |
|---------------|------|
| **Dev** (Fargate, no HA) | ~$90-120/mo |
| **Staging** (GPU, no HA) | ~$750-800/mo |
| **Production** (GPU, HA) | ~$800-850/mo |

> 💡 **Cost tips**:
> - Set `high_availability = false` for dev/staging to save ~$32/month
> - Use `g4dn.xlarge` instead of `g5.xlarge` to save ~$400/month (slower)
> - Set `enable_gpu = false` to use Fargate (much slower but cheaper)

---

## Outputs

After `terraform apply`, you get:

| Output | Description |
|--------|-------------|
| `webhook_url` | URL to configure in GitHub webhooks |
| `health_check_url` | Health check endpoint |
| `alb_dns_name` | ALB DNS name |
| `ecs_cluster_name` | ECS cluster name |
| `ecs_service_name` | ECS service name |
| `ecr_repository_url` | ECR URL for pushing images |
| `cloudwatch_log_group` | Log group for viewing logs |
| `setup_instructions` | Next steps after deployment |

---

## Remote State (Recommended)

For team collaboration, configure S3 backend for remote state:

1. **Create S3 bucket**:
   ```bash
   aws s3 mb s3://your-terraform-state-bucket
   ```

2. **Create DynamoDB table for locking**:
   ```bash
   aws dynamodb create-table \
     --table-name terraform-locks \
     --attribute-definitions AttributeName=LockID,AttributeType=S \
     --key-schema AttributeName=LockID,KeyType=HASH \
     --billing-mode PAY_PER_REQUEST
   ```

3. **Uncomment backend block in `main.tf`** and configure.

---

## Operations

### View Logs

```bash
# Via AWS CLI
aws logs tail /ecs/code-review-production --follow

# Via Makefile
make aws-logs
```

### Update Service

```bash
# After pushing new image
aws ecs update-service \
  --cluster code-review-production-cluster \
  --service code-review-production \
  --force-new-deployment
```

### Scale Up/Down

```bash
terraform apply -var="desired_count=2"
```

### Switch GPU/CPU Mode

```bash
# Enable GPU (faster, more expensive)
terraform apply -var="enable_gpu=true"

# Disable GPU (slower, cheaper)
terraform apply -var="enable_gpu=false"
```

---

## Security

### Network Security

- **Private subnets**: ECS tasks run in private subnets with no public IPs
- **Security groups**: 
  - ALB: Allows 80/443 from anywhere
  - ECS: Only allows traffic from ALB
  - EFS: Only allows NFS from ECS
- **NAT Gateway**: Outbound-only internet access

### Secrets Management

- GitHub token stored in AWS Secrets Manager
- ECS retrieves secret at runtime (never in image)
- Secret ARN passed to task definition

### IAM

- **Execution role**: Minimal permissions for ECS to pull images and secrets
- **Task role**: No additional permissions (add if needed)

---

## HTTPS Setup

For production, add an ACM certificate:

1. **Request certificate** in AWS Certificate Manager
2. **Validate** via DNS or email
3. **Add to terraform.tfvars**:

```hcl
certificate_arn = "arn:aws:acm:us-west-2:123456789:certificate/abc-123"
```

4. **Apply**: `terraform apply`

The ALB will now serve HTTPS and redirect HTTP to HTTPS.

---

## Troubleshooting

### Task keeps restarting

```bash
# Check task logs
aws logs tail /ecs/code-review-production --follow

# Check task status
aws ecs describe-tasks \
  --cluster code-review-production-cluster \
  --tasks $(aws ecs list-tasks --cluster code-review-production-cluster --query 'taskArns[0]' --output text)
```

### Can't reach webhook

1. Check ALB security group allows inbound 80/443
2. Verify target group health checks passing
3. Check ECS service has running tasks
4. If `enable_alb_logs = true`, check S3 bucket for access logs

### Ollama model not loading

1. Check EFS mount is working
2. Verify task has enough memory for model
3. Check Ollama container logs

### GPU instance not starting

1. Check GPU quota in your region
2. Verify AMI is available in your region
3. Check ASG activity in EC2 console

---

## Cleanup

```bash
terraform destroy
```

⚠️ **Warning**: This deletes all resources including:
- EFS (Ollama models)
- Secrets Manager secret
- CloudWatch logs
- S3 ALB logs bucket (if enabled)

---

## CI/CD Integration

The GitHub Actions workflow (`.github/workflows/code-review-agent-deploy.yml`) can deploy automatically:

```yaml
env:
  TF_VAR_github_token: ${{ secrets.TF_VAR_github_token }}

steps:
  - name: Terraform Apply
    run: |
      cd axiompy/agents/code_review/terraform/aws
      terraform init
      terraform apply -auto-approve
```

Required secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `TF_VAR_github_token`

---

## Related Documentation

- [Parent: Code Review Agent](../README.md)
- [Docker Deployment](../docker/)
- [Makefile Commands](../Makefile)
