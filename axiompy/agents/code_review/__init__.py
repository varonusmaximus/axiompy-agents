"""Code Review Agent - AI-powered code review with Clean Architecture.

This module provides code review capabilities that can run as:
- CLI: `axiompy code-review ./src`
- Library: In-process Python API
- Webhook: FastAPI service for GitHub webhooks
- GitHub Action: CI/CD integration

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                 ADAPTERS (Entry Points)              │
    │   CLI  |  Webhook  |  Library  |  GitHub Action     │
    └─────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────┐
    │                APPLICATION (Use Cases)               │
    │              CodeReviewService                       │
    │   Depends on Ports: CodeSource, AIAnalyzer, etc.    │
    └─────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────┐
    │                  DOMAIN (Pure Python)                │
    │   CodeFile, ParsedRule, ReviewResult, RulesEngine   │
    └─────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────┐
    │              INFRASTRUCTURE (Adapters)               │
    │   FileSystem | GitHub | Ollama | OpenAI | Console   │
    └─────────────────────────────────────────────────────┘

Quick Start:
    # CLI
    $ axiompy code-review ./src
    $ axiompy code-review --pr owner/repo#123

    # Library (enum-based factory)
    >>> from axiompy.agents.code_review import (
    ...     CodeReviewServiceFactory,
    ...     CodeSourceType,
    ...     AnalyzerType,
    ... )
    >>> service = CodeReviewServiceFactory.create(
    ...     code_source_type=CodeSourceType.FILESYSTEM,
    ...     analyzer_type=AnalyzerType.OLLAMA,
    ... )
    >>> result = service.review_files(["src/main.py"])
    >>> print(f"Score: {result.score}")

    # Convenience functions
    >>> from axiompy.agents.code_review import review_files, review_pr
    >>> result = review_files(["src/"])
    >>> result = review_pr("owner", "repo", 123)
"""

# Domain layer (pure Python, no dependencies)
# Infrastructure layer (adapters)
from .adapters import (
    # Analyzers (uses axiompy.reasoning under the hood)
    AnalyzerFactory,
    AnalyzerSettings,
    AnalyzerType,
    # Publishers
    ConsolePublisher,
    # Rules
    FileRulesSource,
    # Sources
    FileSystemSource,
    GitHubPublisher,
    GitHubRulesSource,
    GitHubSource,
    GitSource,
    JSONPublisher,
    MockAnalyzer,
    MockCodeSource,
    MockRulesSource,
)

# Convenience functions (library application)
from .applications import (
    create_service,
    review_diff,
    review_files,
    review_pr,
)

# Defaults (single source of truth for configuration)
from .defaults import (
    DEFAULT_CHUNKS,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_RULES_PATH,
    DEFAULT_TIMEOUT_SECS,
)

# Domain also includes ports and service
from .domain import (
    AIAnalyzer,
    CodeExample,
    # Models
    CodeFile,
    # Service
    CodeReviewService,
    # Ports (protocols)
    CodeSource,
    FileDiff,
    # Rules
    ParsedRule,
    PullRequestInfo,
    ReviewComment,
    ReviewPublisher,
    # Results
    ReviewResult,
    ReviewSeverity,
    RuleCategory,
    # Engine
    RulesEngine,
    RuleSeverity,
    RulesSource,
    RuleType,
    Violation,
)

# Factory (dependency injection)
from .factory import (
    CodeReviewServiceFactory,
    CodeReviewSettings,
    CodeSourceSettings,
    CodeSourceType,
    PublisherSettings,
    PublisherType,
    RulesSourceSettings,
    RulesSourceType,
)

__all__ = [
    # === Domain ===
    # Models
    "CodeFile",
    "FileDiff",
    "PullRequestInfo",
    # Rules
    "ParsedRule",
    "RuleType",
    "RuleSeverity",
    "RuleCategory",
    "CodeExample",
    # Results
    "ReviewResult",
    "Violation",
    "ReviewComment",
    "ReviewSeverity",
    # Engine
    "RulesEngine",
    # === Application ===
    # Ports
    "CodeSource",
    "RulesSource",
    "AIAnalyzer",
    "ReviewPublisher",
    # Service
    "CodeReviewService",
    # === Infrastructure ===
    # Sources
    "FileSystemSource",
    "GitHubSource",
    "GitSource",
    "MockCodeSource",
    # Rules
    "FileRulesSource",
    "GitHubRulesSource",
    "MockRulesSource",
    # Analyzers
    "AnalyzerFactory",
    "AnalyzerType",
    "AnalyzerSettings",
    "MockAnalyzer",
    # Publishers
    "ConsolePublisher",
    "GitHubPublisher",
    "JSONPublisher",
    # === Factory ===
    "CodeReviewServiceFactory",
    "CodeReviewSettings",
    # Type enums
    "CodeSourceType",
    "RulesSourceType",
    "PublisherType",
    # Settings
    "CodeSourceSettings",
    "RulesSourceSettings",
    "PublisherSettings",
    # === Convenience Functions ===
    "review_files",
    "review_diff",
    "review_pr",
    "create_service",
    # === Defaults ===
    "DEFAULT_MODEL",
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_TIMEOUT_SECS",
    "DEFAULT_RULES_PATH",
    "DEFAULT_CHUNKS",
]
