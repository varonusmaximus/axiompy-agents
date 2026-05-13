"""
AI Agents for automating development workflows.

Built with Clean Architecture for flexibility and testability:
- CLI: `axiompy code-review ./src`
- Library: In-process Python API
- Webhook: FastAPI service for GitHub webhooks
- GitHub Action: CI/CD integration

Quick Example:
    >>> from axiompy.agents.code_review import CodeReviewServiceFactory
    >>>
    >>> # Review local files
    >>> service = CodeReviewServiceFactory.create_for_filesystem()
    >>> result = service.review_files(["src/main.py"])
    >>> print(f"Score: {result.score}/100")
    >>>
    >>> # Review a GitHub PR (requires GITHUB_TOKEN env var)
    >>> service = CodeReviewServiceFactory.create_from_env()
    >>> result = service.review_pull_request("owner", "repo", 123)

    Note: Never hardcode tokens! Use environment variables.

For comprehensive documentation, see:
    - axiompy/agents/README.md
    - axiompy/agents/code_review/README.md
"""

from axiompy.agents.code_review import (
    AIAnalyzer,
    AnalyzerFactory,
    AnalyzerSettings,
    AnalyzerType,
    # Domain
    CodeFile,
    # Application
    CodeReviewService,
    # Factory
    CodeReviewServiceFactory,
    CodeSource,
    ConsolePublisher,
    FileDiff,
    FileRulesSource,
    # Infrastructure
    FileSystemSource,
    GitHubPublisher,
    GitHubRulesSource,
    GitHubSource,
    GitSource,
    JSONPublisher,
    MockAnalyzer,
    MockCodeSource,
    MockRulesSource,
    ParsedRule,
    PullRequestInfo,
    ReviewComment,
    ReviewPublisher,
    ReviewResult,
    ReviewSeverity,
    RuleCategory,
    RulesEngine,
    RuleSeverity,
    RulesSource,
    RuleType,
    Violation,
    create_service,
    review_diff,
    # Convenience
    review_files,
    review_pr,
)

__all__ = [
    # Domain
    "CodeFile",
    "FileDiff",
    "PullRequestInfo",
    "ParsedRule",
    "RuleType",
    "RuleSeverity",
    "RuleCategory",
    "ReviewResult",
    "Violation",
    "ReviewComment",
    "ReviewSeverity",
    "RulesEngine",
    # Application
    "CodeReviewService",
    "CodeSource",
    "RulesSource",
    "AIAnalyzer",
    "ReviewPublisher",
    # Infrastructure
    "FileSystemSource",
    "GitHubSource",
    "GitSource",
    "MockCodeSource",
    "FileRulesSource",
    "GitHubRulesSource",
    "MockRulesSource",
    "AnalyzerFactory",
    "AnalyzerType",
    "AnalyzerSettings",
    "MockAnalyzer",
    "ConsolePublisher",
    "GitHubPublisher",
    "JSONPublisher",
    # Factory
    "CodeReviewServiceFactory",
    # Convenience
    "review_files",
    "review_diff",
    "review_pr",
    "create_service",
]
