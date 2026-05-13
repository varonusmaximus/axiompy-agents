# Code Review Agent

AI-powered code review that enforces patterns from AGENTS.md.

## Overview

The Code Review Agent automatically reviews code against your organization's coding standards defined in `AGENTS.md`. Built with **Clean Architecture**, it can run as:

- **CLI**: `axiompy code-review ./src` - Review local files
- **Library**: In-process Python API
- **Webhook**: HTTP service for GitHub webhooks (via `axiompy.servers.HTTPServerFactory`)
- **GitHub Action**: Integrated into CI/CD

### Features

- ✅ **Review local files, git changes, or GitHub PRs**
- ✅ **Dynamic rule loading** from AGENTS.md
- ✅ **Multiple AI providers** (Ollama, OpenAI, Anthropic)
- ✅ **Multiple output formats** (console, JSON, GitHub PR comments)
- ✅ **Local LLM** with Ollama - no API keys required
- ✅ **Data privacy** - code stays on your adapters
- ✅ **Clean Architecture** - testable, flexible, extensible

---

## Architecture

This architecture evolved during the development of this branch. The initial implementation was tightly coupled, which became problematic as requirements emerged:

1. **Initial goal**: Build a webhook that reviews GitHub PRs
2. **New requirement**: "Can we also run this as a CLI for local development?"
3. **New requirement**: "Can we review local files without GitHub?"
4. **Testing pain**: Mocking GitHub API for every test was tedious
5. **Realization**: The core review logic shouldn't know about GitHub at all

This led to a refactor toward Clean Architecture, separating *what* the agent does (domain) from *how* it connects to external systems (adapters).

### Before: Tightly Coupled (Initial Implementation)

```
┌────────────────────────────────────────────────────────────────┐
│                 OLD: CodeReviewAgent                            │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────┐   │
│  │GitHubClient │  │ RulesParser │  │      AIClient         │   │
│  │ (hardcoded) │  │   (mixed)   │  │    (hardcoded)        │   │
│  └─────────────┘  └─────────────┘  └───────────────────────┘   │
│                                                                 │
│  Problems discovered during development:                        │
│  ❌ Only works with GitHub PRs                                  │
│  ❌ Can't review local files                                    │
│  ❌ Can't run as CLI                                            │
│  ❌ Hard to test without GitHub                                 │
│  ❌ Locked to specific AI provider                              │
└────────────────────────────────────────────────────────────────┘
```

### After: Clean Architecture (Refactored)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         APPLICATIONS (Executables)                           │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│   │     CLI      │   │   Webhook    │   │GitHub Action │   │   Library    │ │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘ │
│          └──────────────────┴──────────────────┴──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DOMAIN (Core Business Logic)                         │
│                                                                              │
│   CodeReviewService:                         Ports (Protocols):              │
│   + review_files(paths) -> ReviewResult      ┌─────────────────────────────┐│
│   + review_diff(base, head) -> ReviewResult  │CodeSource│RulesSource│AIPort││
│   + review_pr(owner, repo, num) -> Result    └─────────────────────────────┘│
│                                                                              │
│   Domain Models:                                                             │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│   │  CodeFile       │  │  ParsedRule     │  │  ReviewResult               │ │
│   │  FileDiff       │  │  RulesEngine    │  │  Violation                  │ │
│   └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE (Port Implementations)                     │
│   Sources:              Analyzers:            Publishers:                    │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│   │ FileSystem      │  │ Ollama          │  │ Console                     │ │
│   │ GitHub          │  │ OpenAI          │  │ GitHub PR                   │ │
│   │ Git (local)     │  │ Anthropic       │  │ JSON                        │ │
│   └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Architecture Comparison

| Aspect | Before (Coupled) | After (Clean) |
|--------|------------------|---------------|
| **Review local files** | ❌ Not possible | ✅ `FileSystemSource` |
| **CLI support** | ❌ No | ✅ First-class |
| **Test without GitHub** | ❌ Requires mocking | ✅ Pure domain tests |
| **Swap AI provider** | ❌ Code changes | ✅ Config change |
| **Add new output** | ❌ Modify core | ✅ Add publisher |
| **GitHub dependency** | ❌ Always required | ✅ Optional adapter |

### Design Decisions

#### 1. "Applications" not "Adapters" for Entry Points

CLI, webhook, and library are in `applications/` because they are the **executable portions** of the agent — things you actually run. In Hexagonal Architecture, "adapter" typically refers to adapters implementations (like `GitHubSource` adapting `CodeSource`). Entry points aren't adapting anything; they're standalone applications that call the domain.

#### 2. Service and Ports in Domain

`CodeReviewService` and port protocols live in `domain/`, not a separate application layer:

- **The service IS the domain** — it orchestrates the core code review logic
- **Ports define what the domain needs** — they're contracts the domain declares
- **Simpler mental model** — domain = core logic + interfaces, adapters = implementations

Traditional Clean Architecture separates Application from Domain, but that adds complexity without benefit here. The service has no external dependencies — it only knows about ports (abstractions).

#### 3. Infrastructure for Port Implementations

`adapters/` contains implementations of domain ports — the "driven adapters" in Hexagonal terms. They adapt external systems (filesystem, GitHub API, Ollama) to satisfy domain contracts. This separation means:

- Domain stays pure Python with no imports from requests, GitHub libraries, etc.
- Easy to swap implementations (mock for testing, Ollama vs OpenAI for production)
- Clear dependency direction: adapters → domain, never the reverse

---

## Configuration

All defaults are centralized in `defaults.py` — change once, applies everywhere:

```python
# axiompy/agents/code_review/defaults.py
DEFAULT_MODEL = "llama3.2:1b"             # Fastest in practice
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT_SECS = 120
DEFAULT_RULES_PATH = "AGENTS.md"
DEFAULT_CHUNKS = ["anti_patterns", "code_smells", "patterns"]
```

Override via:

```bash
# Environment variables
export OLLAMA_MODEL=mistral
export OLLAMA_HOST=http://gpu-server:11434

# Or programmatically
from axiompy.agents.code_review import (
    CodeReviewServiceFactory,
    CodeSourceType,
    AnalyzerSettings,
    RulesSourceSettings,
)

service = CodeReviewServiceFactory.create(
    code_source_type=CodeSourceType.FILESYSTEM,
    analyzer_settings=AnalyzerSettings(model="mistral"),
    rules_source_settings=RulesSourceSettings(rules_path="custom-rules.md"),
)
```

---

## Quick Start

### Option 1: CLI (Local Files)

```bash
# Review files in current directory
axiompy code-review .

# Review specific files
axiompy code-review src/main.py src/utils.py

# Review staged git changes
axiompy code-review --staged

# Review with JSON output (for CI)
axiompy code-review . --output json

# Review a GitHub PR
axiompy code-review --pr varonusmaximus/axiompy#42
```

### Option 2: Python Library

```python
from axiompy.agents.code_review import (
    CodeReviewServiceFactory,
    CodeSourceType,
    AnalyzerType,
)

# Review local files (enum-based factory)
service = CodeReviewServiceFactory.create(
    code_source_type=CodeSourceType.FILESYSTEM,
    analyzer_type=AnalyzerType.OLLAMA,
)
result = service.review_files(["src/main.py"])

print(f"Score: {result.score}/100")
print(f"Violations: {len(result.violations)}")

for v in result.violations:
    print(f"  {v.file}:{v.line} [{v.severity}] {v.message}")
```

### Option 3: Webhook Service

```bash
cd axiompy/agents/code_review/docker
export GITHUB_TOKEN="ghp_your_token"
docker compose up -d
```

### Option 4: GitHub Action

```yaml
- name: AI Code Review
  run: axiompy code-review --pr ${{ github.repository }}#${{ github.event.pull_request.number }}
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## Directory Structure

```
axiompy/agents/code_review/
│
├── domain/                          # DOMAIN - Core business logic
│   ├── __init__.py
│   ├── models.py                    # CodeFile, FileDiff, PullRequestInfo
│   ├── rules.py                     # ParsedRule, RuleType, RuleSeverity
│   ├── results.py                   # ReviewResult, Violation, ReviewComment
│   ├── engine.py                    # RulesEngine - parsing and prompt building
│   ├── ports.py                     # Interfaces (CodeSource, AIAnalyzer, etc.)
│   └── service.py                   # CodeReviewService
│
├── adapters/                  # INFRASTRUCTURE - Port implementations
│   ├── __init__.py
│   ├── sources/                     # CodeSource implementations
│   │   ├── filesystem.py            # FileSystemSource
│   │   ├── github.py                # GitHubSource
│   │   └── git.py                   # GitSource (local repo)
│   ├── rules/                       # RulesSource implementations
│   │   ├── file.py                  # FileRulesSource
│   │   └── github.py                # GitHubRulesSource
│   ├── analyzers/                   # AIAnalyzer implementations
│   │   └── analyzer.py              # AnalyzerFactory, AnalyzerType, AnalyzerSettings
│   │                                # (Ollama, OpenAI, Anthropic, Mock - via factory)
│   └── publishers/                  # ReviewPublisher implementations
│       ├── console.py               # ConsolePublisher (CLI)
│       ├── github.py                # GitHubPublisher (PR comments)
│       └── json.py                  # JSONPublisher (CI output)
│
├── applications/                    # APPLICATIONS - Executable entry points
│   ├── __init__.py
│   ├── cli.py                       # Command-line application
│   ├── webhook.py                   # HTTP webhook application (uses axiompy.servers)
│   └── library.py                   # Python library API
│
├── __init__.py                      # Public API exports
├── factory.py                       # Dependency injection
├── defaults.py                      # Default configuration (single source of truth)
│
├── docker/                          # Docker deployment
├── terraform/                       # AWS deployment
└── Makefile                         # Build commands
```

---

## Ports (Interfaces)

The service depends only on these abstract protocols, defined in the domain:

```python
# domain/ports.py
from typing import Protocol, List, Optional

class CodeSource(Protocol):
    """Port: How we get code to review."""
    
    def get_files(self, paths: List[str]) -> List[CodeFile]:
        """Get files from paths."""
        ...
    
    def get_diff(self, base: str, head: str) -> List[FileDiff]:
        """Get diff between refs."""
        ...


class RulesSource(Protocol):
    """Port: How we get rules."""
    
    def get_rules(self) -> str:
        """Get AGENTS.md content."""
        ...


class AIAnalyzer(Protocol):
    """Port: How we analyze code."""
    
    def analyze(self, prompt: str) -> str:
        """Send prompt to AI, get response."""
        ...


class ReviewPublisher(Protocol):
    """Port: How we publish results."""
    
    def publish(self, result: ReviewResult, context: dict) -> None:
        """Publish review result."""
        ...
```

---

## Service (Domain Core)

```python
# domain/service.py
@dataclass
class CodeReviewService:
    """Orchestrates code review. Depends only on ports."""
    
    code_source: CodeSource
    rules_source: RulesSource
    analyzer: AIAnalyzer
    publisher: Optional[ReviewPublisher] = None
    
    def review_files(self, paths: List[str]) -> ReviewResult:
        """Review specific files."""
        files = self.code_source.get_files(paths)
        return self._review(files)
    
    def review_diff(self, base: str, head: str) -> ReviewResult:
        """Review git diff."""
        diffs = self.code_source.get_diff(base, head)
        return self._review([CodeFile.from_diff(d) for d in diffs])
    
    def review_pr(self, owner: str, repo: str, pr_number: int) -> ReviewResult:
        """Review GitHub PR."""
        pr = self.code_source.get_pull_request(owner, repo, pr_number)
        result = self._review([CodeFile.from_diff(f) for f in pr.files])
        
        if self.publisher:
            self.publisher.publish(result, {"owner": owner, "repo": repo, "pr": pr_number})
        
        return result
```

---

## Factory (Dependency Injection)

Uses enum-based type selection consistent with other axiompy factories (DatabaseFactory, ObjectStorageFactory):

```python
# factory.py
from enum import Enum

class CodeSourceType(str, Enum):
    FILESYSTEM = "filesystem"
    GIT = "git"
    GITHUB = "github"

class RulesSourceType(str, Enum):
    FILE = "file"
    GITHUB = "github"

class PublisherType(str, Enum):
    CONSOLE = "console"
    JSON = "json"
    GITHUB = "github"
    NONE = "none"

class CodeReviewServiceFactory:
    """Creates service with appropriate adapters using enum-based type selection."""
    
    @staticmethod
    def create(
        code_source_type: CodeSourceType = CodeSourceType.FILESYSTEM,
        rules_source_type: RulesSourceType = RulesSourceType.FILE,
        analyzer_type: AnalyzerType = AnalyzerType.OLLAMA,
        publisher_type: PublisherType = PublisherType.CONSOLE,
        code_source_settings: CodeSourceSettings = None,
        rules_source_settings: RulesSourceSettings = None,
        analyzer_settings: AnalyzerSettings = None,
        publisher_settings: PublisherSettings = None,
    ) -> CodeReviewService:
        """Create service with specified component types."""
        ...
    
    @staticmethod
    def create_mock() -> CodeReviewService:
        """Testing - all mocked."""
        ...
```

**Example usage:**

```python
from axiompy.agents.code_review import (
    CodeReviewServiceFactory,
    CodeSourceType,
    RulesSourceType,
    AnalyzerType,
    PublisherType,
    CodeSourceSettings,
    AnalyzerSettings,
)

# CLI use case - local files
service = CodeReviewServiceFactory.create(
    code_source_type=CodeSourceType.FILESYSTEM,
    analyzer_type=AnalyzerType.OLLAMA,
)

# Webhook use case - GitHub PRs
service = CodeReviewServiceFactory.create(
    code_source_type=CodeSourceType.GITHUB,
    rules_source_type=RulesSourceType.GITHUB,
    publisher_type=PublisherType.GITHUB,
    code_source_settings=CodeSourceSettings(github_token=token),
    rules_source_settings=RulesSourceSettings(
        github_token=token,
        github_repo="varonusmaximus/axiompy",
    ),
    analyzer_settings=AnalyzerSettings(model="qwen2.5-coder:14b"),
    publisher_settings=PublisherSettings(github_token=token),
        )
```

---

## Usage Examples

### CLI Usage

```bash
# Review current directory
axiompy code-review .

# Review with custom rules file
axiompy code-review . --rules ./my-rules.md

# Review staged changes only
axiompy code-review --staged

# Review specific commit
axiompy code-review --diff HEAD~1..HEAD

# JSON output for CI pipelines
axiompy code-review . --output json --fail-on error

# Review GitHub PR (requires GITHUB_TOKEN)
axiompy code-review --pr varonusmaximus/axiompy#42
```

### Library Usage

```python
from axiompy.agents.code_review import (
    CodeReviewServiceFactory,
    CodeSourceType,
    AnalyzerType,
    AnalyzerSettings,
    CodeSourceSettings,
    RulesSourceSettings,
    PublisherType,
    PublisherSettings,
)

# Simple: Review local files with defaults
service = CodeReviewServiceFactory.create(
    code_source_type=CodeSourceType.FILESYSTEM,
)
result = service.review_files(["src/main.py", "src/utils.py"])

# Custom configuration
service = CodeReviewServiceFactory.create(
    code_source_type=CodeSourceType.FILESYSTEM,
    analyzer_type=AnalyzerType.OLLAMA,
    publisher_type=PublisherType.CONSOLE,
    code_source_settings=CodeSourceSettings(root="./src"),
    rules_source_settings=RulesSourceSettings(rules_path="AGENTS.md"),
    analyzer_settings=AnalyzerSettings(model="qwen2.5-coder:14b"),
    publisher_settings=PublisherSettings(verbose=True),
)

# Review and get result
result = service.review_files(["main.py"])

if result.has_errors:
    print("Review failed!")
    for v in result.violations:
        print(f"  {v.file}:{v.line} - {v.message}")
    sys.exit(1)
```

### Chunked Reviews (Recommended for Large Rule Sets)

AGENTS.md can have 100+ rules, creating large prompts that may timeout. Chunked reviews split rules into categories and review each separately:

```python
from axiompy.agents.code_review import CodeReviewServiceFactory, CodeSourceType

service = CodeReviewServiceFactory.create(
    code_source_type=CodeSourceType.FILESYSTEM,
)

# Review in chunks - prevents timeout, allows prioritization
result = service.review_files_chunked(
    ["src/main.py"],
    chunks=["anti_patterns", "code_smells", "patterns"]  # Priority order
)

# Or just check critical rules first
result = service.review_files_chunked(
    ["src/main.py"],
    chunks=["anti_patterns", "code_smells"]  # Skip patterns for speed
)

print(f"Score: {result.score}/100")
print(f"Rules applied: {result.rules_applied}")
```

**Chunks available**:
| Chunk | Count | Description |
|-------|-------|-------------|
| `anti_patterns` | ~8 | God Class, Singleton, etc. (most critical) |
| `code_smells` | ~15 | Magic Numbers, Long Method, etc. |
| `patterns` | ~77 | Factory, Settings, Fluent API, etc. |

### Webhook Usage

```python
import os
from axiompy.agents.code_review import (
    CodeReviewServiceFactory,
    CodeSourceType,
    RulesSourceType,
    PublisherType,
    CodeSourceSettings,
    RulesSourceSettings,
    PublisherSettings,
    AnalyzerSettings,
)
from axiompy.servers import HTTPServerFactory

token = os.environ["GITHUB_TOKEN"]

# Create review service with GitHub adapters (enum-based)
service = CodeReviewServiceFactory.create(
    code_source_type=CodeSourceType.GITHUB,
    rules_source_type=RulesSourceType.GITHUB,
    publisher_type=PublisherType.GITHUB,
    code_source_settings=CodeSourceSettings(github_token=token),
    rules_source_settings=RulesSourceSettings(
        github_token=token,
        github_repo="varonusmaximus/axiompy",
    ),
    analyzer_settings=AnalyzerSettings(model="qwen2.5-coder:14b"),
    publisher_settings=PublisherSettings(github_token=token),
)

# Create webhook handler
def handle_webhook(payload: dict) -> dict:
    if payload.get("action") not in ["opened", "synchronize"]:
        return {"status": "ignored"}
    
    pr = payload["pull_request"]
    owner, repo = payload["repository"]["full_name"].split("/")
    
    result = service.review_pull_request(owner, repo, pr["number"])
    
    return {
        "status": "reviewed",
        "score": result.score,
        "violations": len(result.violations),
    }

# Create and run server using axiompy.servers
server = HTTPServerFactory.create(host="0.0.0.0", port=8000)
server.add_route("/webhook", handle_webhook, methods=["POST"])
server.add_route("/health", lambda: {"status": "healthy"}, methods=["GET"])
server.run()
```

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GITHUB_TOKEN` | For GitHub | - | Token with `repo` scope |
| `OLLAMA_HOST` | No | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | No | `codellama` | Model to use |
| `OPENAI_API_KEY` | For OpenAI | - | OpenAI API key |
| `RULES_PATH` | No | `AGENTS.md` | Path to rules file |

### Supported AI Providers

| Provider | Model | API Key Required | Streaming |
|----------|-------|------------------|-----------|
| Ollama | `llama3.2:1b` (default), `qwen2.5-coder:1.5b`, `mistral` | No | ✅ Yes |
| OpenAI | `gpt-4o`, `gpt-4-turbo` | Yes | No |
| Anthropic | `claude-sonnet-4-20250514` | Yes | No |

### Streaming (Ollama)

Ollama uses streaming by default to prevent timeouts with large prompts. Tokens are received as they're generated instead of waiting for the full response:

```python
from axiompy.agents.code_review.adapters.analyzers import AnalyzerFactory, AnalyzerType, AnalyzerSettings

# Streaming is enabled by default
settings = AnalyzerSettings(model="mistral:latest", stream=True)
analyzer = AnalyzerFactory.create(AnalyzerType.OLLAMA, settings)

# For non-streaming (may timeout on large prompts)
settings = AnalyzerSettings(model="mistral:latest", stream=False)
analyzer = AnalyzerFactory.create(AnalyzerType.OLLAMA, settings)
```

---

## GPU Acceleration

GPU acceleration provides **10-20x faster** inference compared to CPU-only mode. This is critical for production deployments where review latency matters.

### Performance Comparison

| Hardware | Model | Time (2 files, 640 lines) | Speedup |
|----------|-------|---------------------------|---------|
| **CPU only** (local Mac) | qwen2.5-coder:1.5b | ~5 min | 1x |
| **CPU** (c6i.2xlarge) | qwen2.5-coder:1.5b | ~3 min | ~1.5x |
| **GPU** (g5.xlarge, A10G) | qwen2.5-coder:1.5b | ~15-20 sec | **15-20x** |
| **GPU** (A100/H100) | qwen2.5-coder:1.5b | ~5-10 sec | **35-70x** |

### Why GPUs Are Faster

| Factor | CPU | GPU |
|--------|-----|-----|
| Parallelism | 8-16 cores | 5,000-16,000 cores |
| Memory bandwidth | ~50 GB/s | 500-3,000 GB/s |
| Optimized for | Sequential tasks | Matrix operations (LLM) |

### Enabling GPU (AWS)

GPU is **enabled by default** in the Terraform configuration:

```hcl
# terraform/aws/terraform.tfvars
enable_gpu = true                   # Default: true
gpu_instance_type = "g5.xlarge"     # A10G GPU (~$1.00/hr)
ollama_model = "qwen2.5-coder:14b"  # Best quality (recommended)
```

**Instance options**:

| Instance | GPU | VRAM | Cost/hr | Best For |
|----------|-----|------|---------|----------|
| `g4dn.xlarge` | T4 | 16GB | ~$0.50 | Budget/dev |
| `g5.xlarge` | A10G | 24GB | ~$1.00 | **Production (recommended)** |
| `g5.2xlarge` | A10G | 24GB | ~$1.50 | High throughput |
| `p4d.24xlarge` | 8x A100 | 320GB | ~$32.00 | Enterprise scale |

### Disabling GPU

For development or cost savings, you can use CPU-only (Fargate):

```hcl
# terraform/aws/terraform.tfvars
enable_gpu = false  # Use Fargate (serverless, no GPU)
```

### Local Development with GPU

If your local machine has a GPU:

```bash
# Check if Ollama detects GPU
ollama run qwen2.5-coder:1.5b --verbose

# Should show: "using CUDA" or "using Metal"
```

For Macs with Apple Silicon, Ollama automatically uses Metal acceleration.

### Model Selection

| Model | Size | Speed | Quality | GPU Memory | Best For |
|-------|------|-------|---------|------------|----------|
| `qwen2.5-coder:1.5b` | 1.5B | ⚡ Very Fast | Basic | ~1GB | Quick checks |
| `qwen2.5-coder:7b` | 7B | Fast | Good | ~4.5GB | Development |
| `qwen2.5-coder:14b` | 14B | Medium | **Excellent** | ~9GB | **Production (recommended)** |
| `codellama:13b` | 13B | Medium | Very Good | ~8GB | Code-focused |
| `mistral:latest` | 7B | Fast | Good | ~4GB | General purpose |

**Recommendation**: Use `qwen2.5-coder:14b` for production — it detects architectural issues like coupling, SRP violations, and design problems that smaller models miss.

### Model Comparison (Real-World Test)

Tested on the same PR (`ie-examples#4` - Python API with intentional design issues):

| Model | Score | Violations Found | Analysis Quality |
|-------|-------|------------------|------------------|
| `qwen2.5-coder:1.5b` | 100/100 | 0 | ❌ Missed all issues |
| `qwen2.5-coder:7b` | 84/100 | 4 | ⚠️ Found some issues |
| `qwen2.5-coder:14b` | 0/100 | **19** | ✅ Found coupling, SRP, architecture issues |

---

## Testing

Clean Architecture makes testing easy at each layer:

```python
# Domain tests - pure unit tests
def test_rules_engine_parses_patterns():
    engine = RulesEngine()
    rules = engine.parse_rules(SAMPLE_AGENTS_MD)
    assert any(r.name == "Factory Pattern" for r in rules)

# Application tests - mock the ports
def test_service_reviews_files():
    service = CodeReviewService(
        code_source=MockCodeSource(files=[mock_file]),
        rules_source=MockRulesSource(rules=SAMPLE_RULES),
        analyzer=MockAnalyzer(response="No issues"),
        publisher=None,
    )
    result = service.review_files(["test.py"])
    assert result.score == 100

# Integration tests - real adapters
def test_filesystem_source_reads_files():
    source = FileSystemSource(root="./tests/fixtures")
    files = source.get_files(["sample.py"])
    assert len(files) == 1
    assert "def sample" in files[0].content
```

Run tests:

```bash
# Unit tests
pytest tests/test_code_review.py -v

# With coverage
pytest tests/test_code_review.py --cov=axiompy.agents.code_review
```

---

## Rules Enforced

The agent enforces rules from `AGENTS.md`:

| Category | Count | Examples |
|----------|-------|----------|
| Patterns | ~86 | Factory, Settings dataclass, Fluent API |
| Anti-Patterns | ~9 | God Class, Singleton, Speculative Generality |
| Code Smells | ~13 | Long Method, Magic Numbers, Global Variables |

---

## Deployment

### Docker Compose (Local)

```bash
cd docker
export GITHUB_TOKEN="ghp_..."
docker compose up -d
```

### Terraform (AWS with GPU)

GPU-accelerated deployment is the default for production performance:

```bash
cd terraform/aws

# Copy and configure
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars: set github_token, rules_repo

# Deploy
terraform init
terraform plan    # Review changes
terraform apply   # Deploy (~5-10 min)
```

**Default configuration**:
- ✅ GPU enabled (`g5.xlarge` with A10G)
- ✅ Fast model (`qwen2.5-coder:1.5b`)
- ✅ ~20-30 second reviews

**Outputs after deploy**:
```
webhook_url = "http://code-review-production-alb-xxx.us-west-2.elb.amazonaws.com/webhook"
expected_performance = "~20-30 seconds per file (GPU accelerated)"
```

### Cost Estimates

#### AWS Infrastructure (Production Setup)

| Resource | Type | Monthly Cost (24/7) |
|----------|------|---------------------|
| **GPU Instance** | g5.xlarge (A10G) | ~$730 |
| **ALB** | Application Load Balancer | ~$22 |
| **NAT Gateway** | 2x (HA mode) | ~$65 |
| **EFS** | Model storage | ~$5 |
| **CloudWatch** | Logs | ~$5 |
| **ECR** | Container registry | ~$1 |
| **Total** | | **~$828/month** |

#### Cost Optimization Options

| Strategy | Savings | Trade-off |
|----------|---------|-----------|
| **Single NAT Gateway** | -$32/month | Reduced HA |
| **Spot Instances** | -50-70% on GPU | May be interrupted |
| **Scale to Zero** | -90% (when idle) | Cold start latency |
| **Smaller Model** (7B) | Same cost, faster | Lower quality |
| **CPU-only (Fargate)** | -$700/month | 15-20x slower |

#### Per-Review Costs

| Mode | Hardware | Time | Cost/Review |
|------|----------|------|-------------|
| **GPU** (14B model) | g5.xlarge | ~35s | ~$0.01 |
| **GPU** (7B model) | g5.xlarge | ~10s | ~$0.003 |
| **CPU** (Fargate) | 4 vCPU | ~5 min | ~$0.02 |

> 💡 **Tip**: For low-volume usage (<100 reviews/month), consider using the CLI locally instead of running 24/7 infrastructure.

See [terraform/README.md](terraform/README.md) for advanced configuration.

---

## Performance Metrics

Real-world performance data from production deployment (December 2024):

### Review Speed by Model (g5.xlarge, A10G GPU)

| Model | File 1 | File 2 | Total | Tokens Generated |
|-------|--------|--------|-------|------------------|
| `qwen2.5-coder:1.5b` | 4.7s | 0.3s | ~5s | 136 |
| `qwen2.5-coder:7b` | 4.1s | 6.1s | ~10s | 581 |
| `qwen2.5-coder:14b` | 15.7s | 19.7s | ~35s | 1,333 |

### Token Generation Rates

| Model | Tokens/sec | First Token Latency |
|-------|------------|---------------------|
| `qwen2.5-coder:1.5b` | ~190 | 2-3s |
| `qwen2.5-coder:7b` | ~80 | 2-4s |
| `qwen2.5-coder:14b` | ~40 | 3-5s |

### Response Quality vs Speed

```
Quality    ████████████████████ 14B (35s, 19 violations found)
           ████████░░░░░░░░░░░░ 7B  (10s, 4 violations found)
           ██░░░░░░░░░░░░░░░░░░ 1.5B (5s, 0 violations found)
           
Speed      ██████████████████░░ 1.5B
           ████████████░░░░░░░░ 7B
           ██████░░░░░░░░░░░░░░ 14B
```

### Infrastructure Deployment Time

| Component | Time |
|-----------|------|
| Terraform apply (fresh) | ~5 min |
| ECS task startup | ~1-2 min |
| Model download (14B, ~9GB) | ~2-3 min |
| Total cold start | ~8-10 min |

---

## GitHub Webhook Setup

After deploying the service, configure GitHub to send PR events:

### 1. Get Your Webhook URL

After `terraform apply`, get the URL from outputs:
```bash
terraform output webhook_url
# Example: http://code-review-production-alb-xxx.us-east-1.elb.amazonaws.com/webhook
```

### 2. Configure GitHub Webhook

1. Go to your repository → **Settings** → **Webhooks** → **Add webhook**
2. Configure:

| Field | Value |
|-------|-------|
| **Payload URL** | `http://<alb_dns>/webhook` |
| **Content type** | `application/json` |
| **Secret** | *(optional, for signature verification)* |
| **Events** | Select "Let me select individual events" → ✅ **Pull requests** |
| **Active** | ✅ Checked |

3. Click **Add webhook**

### 3. Test the Integration

**Health check:**
```bash
curl http://<alb_dns>/health
# {"status": "healthy", "ollama_model": "qwen2.5-coder:1.5b", ...}
```

**Manual test (simulate PR event):**
```bash
curl -X POST http://<alb_dns>/webhook \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Event: pull_request" \
  -d '{
    "action": "opened",
    "pull_request": {"number": 1},
    "repository": {"name": "your-repo", "owner": {"login": "your-org"}}
  }'
```

**Real test:**
- Open a PR in your repository
- Check the webhook delivery in GitHub (Settings → Webhooks → Recent Deliveries)
- Check CloudWatch logs: `aws logs tail /ecs/code-review-production --follow`

### 4. Webhook Events Handled

| Event | Action | Result |
|-------|--------|--------|
| `pull_request` | `opened` | ✅ Reviews PR |
| `pull_request` | `synchronize` | ✅ Reviews updated PR |
| `pull_request` | `reopened` | ✅ Reviews PR |
| `ping` | - | ✅ Returns `{"status": "pong"}` |
| Other | - | ⏭️ Ignored |

---

## Related Documentation

- [Parent: AxiomPy Agents](../README.md) - Architecture pattern
- [Terraform](terraform/README.md) - Cloud deployment
- [AGENTS.md](../../../AGENTS.md) - Rules enforced
