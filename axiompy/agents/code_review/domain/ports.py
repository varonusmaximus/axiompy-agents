"""Application Ports - Protocol definitions for dependencies.

Ports define the interfaces that the application layer depends on.
Infrastructure adapters implement these protocols.

Using Protocol (structural subtyping) allows any class with matching
methods to satisfy the port, without explicit inheritance.
"""

from typing import List, Optional, Protocol, runtime_checkable

from .models import CodeFile, FileDiff, PullRequestInfo
from .results import ReviewResult


@runtime_checkable
class CodeSource(Protocol):
    """
    Port: How we get code to review.

    Implementations:
    - FileSystemSource: Read from local filesystem
    - GitHubSource: Fetch from GitHub API
    - GitSource: Read from local git repository
    """

    def get_files(self, paths: List[str]) -> List[CodeFile]:
        """
        Get code files from paths.

        Args:
            paths: List of file or directory paths

        Returns:
            List of CodeFile objects
        """
        ...

    def get_file_content(self, path: str) -> str:
        """
        Get content of a single file.

        Args:
            path: Path to the file

        Returns:
            File content as string
        """
        ...

    def get_diff(self, base: str, head: str) -> List[FileDiff]:
        """
        Get diff between two refs (commits, branches).

        Args:
            base: Base ref (e.g., "main", "HEAD~1")
            head: Head ref (e.g., "feature-branch", "HEAD")

        Returns:
            List of FileDiff objects
        """
        ...

    def get_pull_request(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> PullRequestInfo:
        """
        Get pull request information.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: PR number

        Returns:
            PullRequestInfo with files

        Note:
            This method is optional - only GitHubSource implements it.
            FileSystemSource and GitSource raise NotImplementedError.
        """
        ...


@runtime_checkable
class RulesSource(Protocol):
    """
    Port: How we get review rules.

    Implementations:
    - FileRulesSource: Read AGENTS.md from filesystem
    - GitHubRulesSource: Fetch AGENTS.md from GitHub repository
    """

    def get_rules(self) -> str:
        """
        Get the main rules content (AGENTS.md).

        Returns:
            Rules content as string (markdown)
        """
        ...

    def get_local_overrides(self) -> Optional[str]:
        """
        Get local rule overrides (.cursorrules).

        Returns:
            Override content if exists, None otherwise
        """
        ...


@runtime_checkable
class AIAnalyzer(Protocol):
    """
    Port: How we analyze code with AI.

    Implementations:
    - OllamaAnalyzer: Local Ollama server
    - OpenAIAnalyzer: OpenAI API
    - AnthropicAnalyzer: Anthropic API
    - MockAnalyzer: For testing
    """

    def analyze(self, prompt: str) -> str:
        """
        Send prompt to AI and get response.

        Args:
            prompt: The full prompt including code and rules

        Returns:
            AI response as string
        """
        ...


@runtime_checkable
class ReviewPublisher(Protocol):
    """
    Port: How we publish review results.

    Implementations:
    - ConsolePublisher: Print to terminal (CLI)
    - GitHubPublisher: Post PR review comments
    - JSONPublisher: Output JSON (CI pipelines)
    """

    def publish(self, result: ReviewResult, context: dict) -> None:
        """
        Publish review result.

        Args:
            result: The review result to publish
            context: Additional context (owner, repo, pr_number, etc.)
        """
        ...
