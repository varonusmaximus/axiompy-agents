"""Factory - Dependency injection for CodeReviewService.

Provides factory method to create properly configured services
using enum-based type selection for sources, analyzers, and publishers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .adapters.analyzers import (
    AnalyzerFactory,
    AnalyzerSettings,
    AnalyzerType,
)
from .adapters.publishers import (
    ConsolePublisher,
    GitHubPublisher,
    JSONPublisher,
)
from .adapters.rules import (
    FileRulesSource,
    GitHubRulesSource,
    MockRulesSource,
)
from .adapters.sources import (
    FileSystemSource,
    GitHubSource,
    GitSource,
    MockCodeSource,
)
from .defaults import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST
from .domain.service import CodeReviewService


class CodeSourceType(str, Enum):
    """Type of code source for the review service."""

    FILESYSTEM = "filesystem"
    GIT = "git"
    GITHUB = "github"


class RulesSourceType(str, Enum):
    """Type of rules source for the review service."""

    FILE = "file"
    GITHUB = "github"


class PublisherType(str, Enum):
    """Type of publisher for review output."""

    CONSOLE = "console"
    JSON = "json"
    GITHUB = "github"
    NONE = "none"


@dataclass
class CodeSourceSettings:
    """Configuration for code sources.

    Attributes:
        root: Root directory for filesystem source
        repo_path: Path to git repository for git source
        github_token: GitHub token for GitHub source
    """

    root: str = "."
    repo_path: str = "."
    github_token: Optional[str] = None


@dataclass
class RulesSourceSettings:
    """Configuration for rules sources.

    Attributes:
        rules_path: Local path to rules file (for FILE source)
        github_token: GitHub token (for GITHUB source)
        github_repo: Repository containing rules, e.g. "owner/repo" (for GITHUB source)
        github_file: Path to rules file in repo (for GITHUB source)
    """

    rules_path: str = "AGENTS.md"
    github_token: Optional[str] = None
    github_repo: str = "varonusmaximus/axiompy"
    github_file: str = "AGENTS.md"


@dataclass
class PublisherSettings:
    """Configuration for publishers.

    Attributes:
        verbose: Verbose console output (for CONSOLE publisher)
        github_token: GitHub token (for GITHUB publisher)
    """

    verbose: bool = False
    github_token: Optional[str] = None


@dataclass
class CodeReviewSettings:
    """Unified configuration for CodeReviewService.

    Attributes:
        code_source_type: Type of code source
        rules_source_type: Type of rules source
        analyzer_type: Type of AI analyzer
        publisher_type: Type of output publisher
        code_source: Code source configuration
        rules_source: Rules source configuration
        analyzer: Analyzer configuration
        publisher: Publisher configuration
    """

    # Type selections
    code_source_type: CodeSourceType = CodeSourceType.FILESYSTEM
    rules_source_type: RulesSourceType = RulesSourceType.FILE
    analyzer_type: AnalyzerType = AnalyzerType.OLLAMA
    publisher_type: PublisherType = PublisherType.CONSOLE

    # Settings for each component
    code_source: CodeSourceSettings = field(default_factory=CodeSourceSettings)
    rules_source: RulesSourceSettings = field(default_factory=RulesSourceSettings)
    analyzer: AnalyzerSettings = field(default_factory=AnalyzerSettings)
    publisher: PublisherSettings = field(default_factory=PublisherSettings)


class CodeReviewServiceFactory:
    """
    Factory for creating CodeReviewService instances.

    Uses enum-based type selection consistent with other axiompy factories
    (DatabaseFactory, ObjectStorageFactory, ReasoningFactory).

    Example:
        # Simple: Review local files with defaults
        service = CodeReviewServiceFactory.create(
            code_source_type=CodeSourceType.FILESYSTEM,
        )
        result = service.review_files(["src/main.py"])

        # Custom: Review GitHub PR with specific model
        service = CodeReviewServiceFactory.create(
            code_source_type=CodeSourceType.GITHUB,
            rules_source_type=RulesSourceType.GITHUB,
            analyzer_type=AnalyzerType.OLLAMA,
            publisher_type=PublisherType.GITHUB,
            code_source_settings=CodeSourceSettings(github_token=token),
            rules_source_settings=RulesSourceSettings(
                github_token=token,
                github_repo="varonusmaximus/axiompy",
            ),
            analyzer_settings=AnalyzerSettings(model="qwen2.5-coder:14b"),
            publisher_settings=PublisherSettings(github_token=token),
        )

        # Using unified settings object
        settings = CodeReviewSettings(
            code_source_type=CodeSourceType.FILESYSTEM,
            analyzer_type=AnalyzerType.OPENAI,
            analyzer=AnalyzerSettings(api_key="sk-...", model="gpt-4o"),
        )
        service = CodeReviewServiceFactory.create_from_settings(settings)
    """

    @staticmethod
    def create(
        code_source_type: CodeSourceType = CodeSourceType.FILESYSTEM,
        rules_source_type: RulesSourceType = RulesSourceType.FILE,
        analyzer_type: AnalyzerType = AnalyzerType.OLLAMA,
        publisher_type: PublisherType = PublisherType.CONSOLE,
        code_source_settings: Optional[CodeSourceSettings] = None,
        rules_source_settings: Optional[RulesSourceSettings] = None,
        analyzer_settings: Optional[AnalyzerSettings] = None,
        publisher_settings: Optional[PublisherSettings] = None,
    ) -> CodeReviewService:
        """
        Create a CodeReviewService with specified component types.

        Args:
            code_source_type: Type of code source (FILESYSTEM, GIT, GITHUB)
            rules_source_type: Type of rules source (FILE, GITHUB)
            analyzer_type: Type of AI analyzer (OLLAMA, OPENAI, ANTHROPIC)
            publisher_type: Type of output publisher (CONSOLE, JSON, GITHUB, NONE)
            code_source_settings: Configuration for code source
            rules_source_settings: Configuration for rules source
            analyzer_settings: Configuration for analyzer
            publisher_settings: Configuration for publisher

        Returns:
            Configured CodeReviewService

        Raises:
            ValueError: If required settings are missing for selected types
        """
        # Apply defaults
        code_settings = code_source_settings or CodeSourceSettings()
        rules_settings = rules_source_settings or RulesSourceSettings()
        ai_settings = analyzer_settings or AnalyzerSettings(
            model=DEFAULT_MODEL,
            host=DEFAULT_OLLAMA_HOST,
        )
        pub_settings = publisher_settings or PublisherSettings()

        # Create code source
        code_source = _create_code_source(code_source_type, code_settings)

        # Create rules source
        rules_source = _create_rules_source(rules_source_type, rules_settings)

        # Create analyzer
        analyzer = AnalyzerFactory.create(analyzer_type, ai_settings)

        # Create publisher
        publisher = _create_publisher(publisher_type, pub_settings)

        return CodeReviewService(
            code_source=code_source,
            rules_source=rules_source,
            analyzer=analyzer,
            publisher=publisher,
        )

    @staticmethod
    def create_from_settings(settings: CodeReviewSettings) -> CodeReviewService:
        """
        Create service from a unified settings object.

        Args:
            settings: Complete configuration for the service

        Returns:
            Configured CodeReviewService
        """
        return CodeReviewServiceFactory.create(
            code_source_type=settings.code_source_type,
            rules_source_type=settings.rules_source_type,
            analyzer_type=settings.analyzer_type,
            publisher_type=settings.publisher_type,
            code_source_settings=settings.code_source,
            rules_source_settings=settings.rules_source,
            analyzer_settings=settings.analyzer,
            publisher_settings=settings.publisher,
        )

    @staticmethod
    def create_mock(
        rules: Optional[str] = None,
        response: Optional[str] = None,
    ) -> CodeReviewService:
        """
        Create mock service for testing.

        Args:
            rules: Mock rules content
            response: Mock AI response

        Returns:
            Mock CodeReviewService
        """
        code_source = MockCodeSource()
        rules_source = MockRulesSource(rules=rules)
        analyzer = AnalyzerFactory.create_mock(response=response)

        return CodeReviewService(
            code_source=code_source,
            rules_source=rules_source,
            analyzer=analyzer,
            publisher=None,
        )


def _create_code_source(source_type: CodeSourceType, settings: CodeSourceSettings):
    """Create code source based on type."""
    match source_type:
        case CodeSourceType.FILESYSTEM:
            return FileSystemSource(root=settings.root)

        case CodeSourceType.GIT:
            return GitSource(repo_path=settings.repo_path)

        case CodeSourceType.GITHUB:
            if not settings.github_token:
                raise ValueError("github_token required for GitHub code source")
            return GitHubSource(token=settings.github_token)

        case _:
            raise ValueError(f"Unknown code source type: {source_type}")


def _create_rules_source(source_type: RulesSourceType, settings: RulesSourceSettings):
    """Create rules source based on type."""
    match source_type:
        case RulesSourceType.FILE:
            return FileRulesSource(rules_path=settings.rules_path)

        case RulesSourceType.GITHUB:
            if not settings.github_token:
                raise ValueError("github_token required for GitHub rules source")
            return GitHubRulesSource(
                token=settings.github_token,
                repo=settings.github_repo,
                rules_file=settings.github_file,
            )

        case _:
            raise ValueError(f"Unknown rules source type: {source_type}")


def _create_publisher(publisher_type: PublisherType, settings: PublisherSettings):
    """Create publisher based on type."""
    match publisher_type:
        case PublisherType.CONSOLE:
            return ConsolePublisher(verbose=settings.verbose)

        case PublisherType.JSON:
            return JSONPublisher()

        case PublisherType.GITHUB:
            if not settings.github_token:
                raise ValueError("github_token required for GitHub publisher")
            return GitHubPublisher(token=settings.github_token)

        case PublisherType.NONE:
            return None

        case _:
            raise ValueError(f"Unknown publisher type: {publisher_type}")
