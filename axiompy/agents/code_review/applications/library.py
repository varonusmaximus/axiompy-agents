"""Library Adapter - Python API for in-process use.

Provides simple functions for using code review as a library.
"""

import os
from typing import List, Optional

from ..adapters.analyzers import AnalyzerSettings, AnalyzerType
from ..domain.results import ReviewResult
from ..factory import (
    CodeReviewServiceFactory,
    CodeSourceSettings,
    CodeSourceType,
    PublisherSettings,
    PublisherType,
    RulesSourceSettings,
    RulesSourceType,
)


def create_service(
    source_type: CodeSourceType = CodeSourceType.FILESYSTEM,
    rules_type: RulesSourceType = RulesSourceType.FILE,
    analyzer_type: AnalyzerType = AnalyzerType.OLLAMA,
    publisher_type: PublisherType = PublisherType.NONE,
    # Source settings
    root: str = ".",
    repo_path: str = ".",
    github_token: Optional[str] = None,
    # Rules settings
    rules_path: str = "AGENTS.md",
    rules_repo: str = "varonusmaximus/axiompy",
    rules_file: str = "AGENTS.md",
    # Analyzer settings
    analyzer_settings: Optional[AnalyzerSettings] = None,
    # Publisher settings
    verbose: bool = False,
):
    """
    Create a CodeReviewService with specified adapters.

    Args:
        source_type: CodeSourceType enum (FILESYSTEM, GITHUB, GIT)
        rules_type: RulesSourceType enum (FILE, GITHUB)
        analyzer_type: AnalyzerType enum (OLLAMA, OPENAI, ANTHROPIC)
        publisher_type: PublisherType enum (CONSOLE, JSON, GITHUB, NONE)
        root: Root directory for filesystem source
        repo_path: Path to git repository
        github_token: GitHub token (or use GITHUB_TOKEN env var)
        rules_path: Path to rules file
        rules_repo: Repository containing rules (owner/repo)
        rules_file: Path to rules file in repo
        analyzer_settings: Pre-configured AnalyzerSettings (uses defaults if not provided)
        verbose: Verbose console output

    Returns:
        Configured CodeReviewService

    Example:
        service = create_service(
            source_type=CodeSourceType.FILESYSTEM,
            rules_type=RulesSourceType.FILE,
            analyzer_type=AnalyzerType.OLLAMA,
        )
    """
    # Resolve GitHub token from env if not provided
    token = github_token or os.environ.get("GITHUB_TOKEN")

    # Build settings objects
    code_source_settings = CodeSourceSettings(
        root=root,
        repo_path=repo_path,
        github_token=token,
    )

    rules_source_settings = RulesSourceSettings(
        rules_path=rules_path,
        github_token=token,
        github_repo=rules_repo,
        github_file=rules_file,
    )

    # Handle analyzer settings with API key resolution
    if analyzer_settings is None:
        analyzer_settings = AnalyzerSettings()

    # Resolve API keys from environment for OpenAI/Anthropic
    if analyzer_type == AnalyzerType.OPENAI and not analyzer_settings.api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required (set OPENAI_API_KEY)")
        analyzer_settings = AnalyzerSettings(
            api_key=api_key,
            model=analyzer_settings.model or "gpt-4o",
        )
    elif analyzer_type == AnalyzerType.ANTHROPIC and not analyzer_settings.api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required (set ANTHROPIC_API_KEY)")
        analyzer_settings = AnalyzerSettings(
            api_key=api_key,
            model=analyzer_settings.model or "claude-sonnet-4-20250514",
        )

    publisher_settings = PublisherSettings(
        verbose=verbose,
        github_token=token,
    )

    return CodeReviewServiceFactory.create(
        code_source_type=source_type,
        rules_source_type=rules_type,
        analyzer_type=analyzer_type,
        publisher_type=publisher_type,
        code_source_settings=code_source_settings,
        rules_source_settings=rules_source_settings,
        analyzer_settings=analyzer_settings,
        publisher_settings=publisher_settings,
    )


def review_files(
    paths: List[str],
    rules_path: str = "AGENTS.md",
    analyzer_type: AnalyzerType = AnalyzerType.OLLAMA,
    analyzer_settings: Optional[AnalyzerSettings] = None,
) -> ReviewResult:
    """
    Review local files.

    Convenience function for quick file review.

    Args:
        paths: List of file or directory paths
        rules_path: Path to rules file
        analyzer_type: AI analyzer to use
        analyzer_settings: Analyzer configuration

    Returns:
        ReviewResult

    Example:
        result = review_files(["src/main.py", "src/utils.py"])
        print(f"Score: {result.score}")
    """
    service = create_service(
        source_type=CodeSourceType.FILESYSTEM,
        rules_type=RulesSourceType.FILE,
        rules_path=rules_path,
        analyzer_type=analyzer_type,
        analyzer_settings=analyzer_settings,
    )
    return service.review_files(paths)


def review_diff(
    base: str = "HEAD~1",
    head: str = "HEAD",
    rules_path: str = "AGENTS.md",
    analyzer_type: AnalyzerType = AnalyzerType.OLLAMA,
    analyzer_settings: Optional[AnalyzerSettings] = None,
) -> ReviewResult:
    """
    Review git diff.

    Convenience function for reviewing git changes.

    Args:
        base: Base ref (default: HEAD~1)
        head: Head ref (default: HEAD)
        rules_path: Path to rules file
        analyzer_type: AI analyzer to use
        analyzer_settings: Analyzer configuration

    Returns:
        ReviewResult

    Example:
        # Review last commit
        result = review_diff("HEAD~1", "HEAD")

        # Review staged changes
        result = review_diff("HEAD", "staged")
    """
    service = create_service(
        source_type=CodeSourceType.GIT,
        rules_type=RulesSourceType.FILE,
        rules_path=rules_path,
        analyzer_type=analyzer_type,
        analyzer_settings=analyzer_settings,
    )
    return service.review_diff(base, head)


def review_pr(
    owner: str,
    repo: str,
    pr_number: int,
    github_token: Optional[str] = None,
    rules_repo: str = "varonusmaximus/axiompy",
    post_review: bool = False,
    analyzer_type: AnalyzerType = AnalyzerType.OLLAMA,
    analyzer_settings: Optional[AnalyzerSettings] = None,
) -> ReviewResult:
    """
    Review a GitHub pull request.

    Convenience function for PR review.

    Args:
        owner: Repository owner
        repo: Repository name
        pr_number: PR number
        github_token: GitHub token (or GITHUB_TOKEN env var)
        rules_repo: Repository containing AGENTS.md
        post_review: Whether to post review to PR
        analyzer_type: AI analyzer to use
        analyzer_settings: Analyzer configuration

    Returns:
        ReviewResult

    Example:
        result = review_pr("owner", "repo", 123)
        if result.has_errors:
            print("PR has issues!")
    """
    token = github_token or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("GitHub token required")

    service = create_service(
        source_type=CodeSourceType.GITHUB,
        rules_type=RulesSourceType.GITHUB,
        analyzer_type=analyzer_type,
        publisher_type=PublisherType.GITHUB if post_review else PublisherType.NONE,
        github_token=token,
        rules_repo=rules_repo,
        analyzer_settings=analyzer_settings,
    )

    return service.review_pull_request(owner, repo, pr_number)
