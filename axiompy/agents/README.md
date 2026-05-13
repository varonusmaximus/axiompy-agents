# AxiomPy Agents

> **📋 Migration Note**: Agent implementations (like `code_review`) are planned to move to
> [ie-examples](https://github.com/varonusmaximus/ie-examples) repository. The `axiompy.agents`
> module will retain only the base patterns, protocols, and factory abstractions that agents
> build upon. The models in `axiompy/agents/models/` are temporary and will also move with
> the implementations.

AI-powered agents for automating development workflows, built with Clean Architecture.

## Overview

The `axiompy.agents` module provides intelligent agents that automate common development tasks using AI. All agents follow **Clean Architecture** (Hexagonal/Ports & Adapters) to ensure:

- **Testability**: Core logic is pure Python with no external dependencies
- **Flexibility**: Same agent works via CLI, webhook, library, or GitHub Action
- **Extensibility**: Add new sources, analyzers, or publishers without modifying core
- **Maintainability**: Clear separation of concerns with well-defined layers

### Available Agents

| Agent | Description | Status |
|-------|-------------|--------|
| [`code_review`](code_review/README.md) | AI-powered code review with AGENTS.md pattern enforcement | ✅ Available |
| [`rag`](rag/README.md) | Local-first RAG for knowledge-grounded AI with multiple embedders & vector stores | ✅ Available |

---

## Architecture Pattern

All agents in `axiompy.agents` follow Clean Architecture. This pattern emerged from building the first agent (code_review) where we discovered that tightly coupling to GitHub made the agent:
- Impossible to run locally as a CLI
- Hard to test without mocking external APIs
- Locked to a single AI provider

By separating the **domain** (what the agent does) from **adapters** (how it connects to external systems), agents become flexible, testable, and reusable across different contexts.

### Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         APPLICATIONS (Executables)                           │
│                                                                              │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│   │     CLI      │   │   Webhook    │   │GitHub Action │   │   Library    │ │
│   │              │   │   (FastAPI)  │   │   Runner     │   │   (in-proc)  │ │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘ │
│          └──────────────────┴──────────────────┴──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DOMAIN (Core)                                   │
│                                                                              │
│   Pure Python. No external dependencies. Fully unit testable.                │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         AgentService                                 │   │
│   │                                                                      │   │
│   │  Orchestrates domain logic. Depends ONLY on Ports (protocols).       │   │
│   │  No knowledge of HTTP, GitHub, CLI, or any external system.          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Ports (Protocols):                                                         │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│   │ InputSource     │  │ Analyzer        │  │ OutputPublisher             │ │
│   │ (Protocol)      │  │ (Protocol)      │  │ (Protocol)                  │ │
│   └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
│                                                                              │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│   │  Domain Models  │  │  Business Rules │  │  Result Types               │ │
│   │  (dataclasses)  │  │  (pure logic)   │  │  (value objects)            │ │
│   └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INFRASTRUCTURE (Port Implementations)                 │
│                                                                              │
│   Implements Ports. Contains all external dependencies.                      │
│                                                                              │
│   Sources:              Analyzers:            Publishers:                    │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│   │ FileSystem      │  │ Ollama          │  │ Console                     │ │
│   │ GitHub          │  │ OpenAI          │  │ GitHub                      │ │
│   │ Git (local)     │  │ Anthropic       │  │ JSON (CI output)            │ │
│   └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Clean Architecture?

| Problem with Tight Coupling | Solution with Clean Architecture |
|----------------------------|----------------------------------|
| Can only run as webhook | CLI, webhook, library, action - same core |
| Hard to test (needs GitHub) | Domain layer is pure, fully testable |
| Can't review local files | FileSystem source adapter |
| Locked to one AI provider | Swap Ollama/OpenAI/Anthropic via config |
| Output only to GitHub | Console, JSON, GitHub publishers |

---

## Design Decisions & Motivations

### 1. "Applications" not "Adapters" for Entry Points

**Decision**: CLI, webhook, and library entry points are in `applications/`, not `adapters/`.

**Motivation**: These are the *executable portions* of the agent — the things you actually run. The term "adapter" in Hexagonal Architecture typically refers to adapters implementations of ports (like `GitHubSource` adapting the `CodeSource` port). But CLI and webhook aren't adapting anything — they're standalone applications that *use* the domain service.

```
applications/     ← Things you RUN (CLI, webhook, library API)
adapters/   ← Implementations of ports (adapters in Hexagonal terms)
```

### 2. Service and Ports in Domain, not Application Layer

**Decision**: `AgentService` and port definitions (protocols) live in `domain/`, not a separate `application/` layer.

**Motivation**: 
- The service IS the domain — it orchestrates the core business logic
- Ports define what the domain *needs*, not how external systems work
- Separating into `domain/` and `application/` added complexity without benefit
- Simpler mental model: domain = core logic + interfaces, adapters = implementations

Traditional Clean Architecture has Application as a separate layer, but for most agents, this separation is unnecessary. The service coordinates domain logic and depends on ports (abstractions) — that's domain behavior.

### 3. Infrastructure Contains Port Implementations

**Decision**: `adapters/` contains implementations of domain ports (sources, analyzers, publishers).

**Motivation**: These are the "driven adapters" in Hexagonal terms — they adapt external systems (filesystem, GitHub, Ollama) to the ports defined by the domain. Keeping them separate from domain means:
- Domain stays pure Python with no external dependencies
- Easy to swap implementations (mock for testing, Ollama vs OpenAI for production)
- Clear dependency direction: adapters depends on domain, never the reverse

---

## Standard Agent Structure

Every agent follows this directory structure:

```
axiompy/agents/{agent_name}/
│
├── domain/                          # DOMAIN - Core business logic
│   ├── __init__.py
│   ├── models.py                    # Domain entities and value objects
│   ├── rules.py                     # Business rules and logic
│   ├── engine.py                    # Core processing engine
│   ├── ports.py                     # Protocol definitions (interfaces)
│   └── service.py                   # Main service orchestrating domain logic
│
├── adapters/                  # INFRASTRUCTURE - Port implementations
│   ├── __init__.py
│   ├── sources/                     # Input source implementations
│   ├── analyzers/                   # AI/analysis implementations
│   └── publishers/                  # Output publisher implementations
│
├── applications/                    # APPLICATIONS - Executable entry points
│   ├── __init__.py
│   ├── cli.py                       # Command-line application
│   ├── webhook.py                   # HTTP webhook application
│   └── library.py                   # In-process Python API
│
├── __init__.py                      # Public API exports
├── factory.py                       # Dependency injection / wiring
│
├── docker/                          # Deployment configs
├── terraform/                       # Infrastructure as code
└── Makefile                         # Build/deploy commands
```

---

## Usage Patterns

### Pattern 1: CLI (Local Development)

```bash
# Review local files
axiompy code-review ./src

# Review staged git changes  
axiompy code-review --staged

# Review a specific commit range
axiompy code-review --diff HEAD~3..HEAD

# Output as JSON for CI
axiompy code-review ./src --output json
```

### Pattern 2: In-Process Library

```python
from axiompy.agents.code_review import CodeReviewServiceFactory

# Quick start - review local files
service = CodeReviewServiceFactory.create_for_filesystem()
result = service.review_files(["src/main.py"])

print(f"Score: {result.score}/100")
for v in result.violations:
    print(f"  {v.file}:{v.line} - {v.message}")
```

### Pattern 3: Webhook (Production Service)

```python
from axiompy.agents.code_review import CodeReviewServiceFactory

# Create service with GitHub adapters
service = CodeReviewServiceFactory.create_for_github(
    token=os.environ["GITHUB_TOKEN"],
    rules_repo="varonusmaximus/axiompy",
)

# Handle webhook
@app.post("/webhook")
async def handle(payload: dict):
    result = service.review_pull_request(
        owner, repo, payload["pull_request"]["number"]
    )
    return {"score": result.score}
```

### Pattern 4: GitHub Action

```yaml
- name: AI Code Review
  run: |
    axiompy code-review \
      --pr ${{ github.repository }}#${{ github.event.pull_request.number }} \
      --output github
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## Dependency Injection

Agents use the **Factory Pattern** for dependency injection:

```python
# Factory creates service with correct adapters for each use case
class AgentServiceFactory:
    
    @staticmethod
    def create_for_filesystem(...) -> AgentService:
        """CLI use case - local files, console output."""
        return AgentService(
            source=FileSystemSource(...),
            analyzer=OllamaAnalyzer(...),
            publisher=ConsolePublisher(),
        )
    
    @staticmethod
    def create_for_github(...) -> AgentService:
        """Webhook use case - GitHub source and publisher."""
        return AgentService(
            source=GitHubSource(...),
            analyzer=OllamaAnalyzer(...),
            publisher=GitHubPublisher(...),
        )
    
    @staticmethod
    def create_mock() -> AgentService:
        """Testing - all mocked adapters."""
        return AgentService(
            source=MockSource(),
            analyzer=MockAnalyzer(),
            publisher=None,
        )
```

---

## Port Definitions

Ports are defined as Python `Protocol` classes:

```python
from typing import Protocol, List

class InputSource(Protocol):
    """Port: How we get data to process."""
    def get_items(self, paths: List[str]) -> List[Item]: ...

class Analyzer(Protocol):
    """Port: How we analyze data."""
    def analyze(self, content: str, context: dict) -> AnalysisResult: ...

class OutputPublisher(Protocol):
    """Port: How we publish results."""
    def publish(self, result: Result, context: dict) -> None: ...
```

Infrastructure adapters implement these protocols:

```python
class FileSystemSource:
    """Implements InputSource for local filesystem."""
    def get_items(self, paths: List[str]) -> List[Item]:
        # Read from filesystem
        ...

class GitHubSource:
    """Implements InputSource for GitHub API."""
    def get_items(self, paths: List[str]) -> List[Item]:
        # Fetch from GitHub
        ...
```

---

## HTTP Client Usage

Infrastructure adapters that make HTTP calls should use `axiompy.io.http.HTTPClientFactory`:

```python
from axiompy.io.http import HTTPClientFactory

class MyAPISource:
    def __init__(self, api_key: str, timeout: int = 30):
        # Use axiompy HTTPClient - consistent retry, logging, auth
        self._http_client = (
            HTTPClientFactory.create(timeout_secs=timeout)
            .bearer_token(api_key)
            .add_header("Content-Type", "application/json")
        )
    
    def get_items(self, paths: List[str]) -> List[Item]:
        response = self._http_client.get(f"{self.base_url}/items")
        return [Item.from_dict(d) for d in response.json()]
```

See [AGENTS.md - HTTPClient Pattern](../../AGENTS.md) for details.

---

## Testing Strategy

Clean Architecture enables testing at each layer:

```python
# 1. Domain tests - pure unit tests, no mocks needed
def test_rules_engine_parses_patterns():
    engine = RulesEngine()
    rules = engine.parse_rules("## Factory Pattern\n...")
    assert len(rules) == 1
    assert rules[0].name == "Factory Pattern"

# 2. Application tests - mock the ports
def test_service_reviews_files():
    service = AgentService(
        source=MockSource(files=[mock_file]),
        analyzer=MockAnalyzer(response="No issues"),
        publisher=None,
    )
    result = service.review_files(["test.py"])
    assert result.score == 100

# 3. Integration tests - real adapters, test environment
def test_github_source_fetches_pr():
    source = GitHubSource(token=TEST_TOKEN)
    pr = source.get_pull_request("owner", "repo", 1)
    assert pr.number == 1
```

---

## Creating a New Agent

1. **Define domain models** in `domain/models.py`
2. **Implement business logic** in `domain/engine.py`
3. **Define ports** in `domain/ports.py`
4. **Create service** in `domain/service.py`
5. **Implement adapters** in `adapters/` (sources, analyzers, publishers)
6. **Add applications** in `applications/` (CLI, webhook, library)
7. **Wire with factory** in `factory.py`

Follow the [Code Review Agent](code_review/README.md) as a reference implementation.

---

## Module Structure

```
axiompy/agents/
├── __init__.py                      # Public API
├── README.md                        # This file
├── code_review/                     # Code Review Agent
│   ├── domain/                      # Core logic, service, ports
│   ├── adapters/                    # Port implementations
│   ├── applications/                # Executables (CLI, webhook, library)
│   ├── docker/                      # Deployment
│   ├── terraform/                   # Infrastructure
│   └── README.md                    # Agent-specific docs
└── rag/                             # RAG Agent (Local-First)
    ├── domain/                      # Models, ports, chunker, service
    ├── adapters/                    # Embedders, vector stores, sources, LLM
    │   ├── embedders/               # FastEmbed, sentence-transformers, Ollama, OpenAI
    │   ├── vector_stores/           # Memory, Chroma, Pinecone, pgvector
    │   ├── sources/                 # FileSystemSource
    │   └── llm/                     # ReasoningAdapter (wraps axiompy.reasoning)
    ├── applications/                # CLI
    ├── factory.py                   # RAGServiceFactory
    └── README.md                    # RAG-specific docs
```

---

## Related Documentation

- [Code Review Agent](code_review/README.md) - Reference implementation for Clean Architecture
- [RAG Agent](rag/README.md) - Local-first RAG with multiple embedders & vector stores
- [AGENTS.md](../../AGENTS.md) - Coding standards enforced by agents
- [axiompy.reasoning](../reasoning/) - AI provider abstraction (used by both agents)
