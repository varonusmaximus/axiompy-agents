"""Domain Models - Core entities for code review.

These models represent the core domain concepts:
- CodeFile: A file with content to review
- FileDiff: Changes to a file (additions, deletions, patch)
- PullRequestInfo: Metadata about a pull request

All models are pure dataclasses with no external dependencies.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CodeFile:
    """
    A file with content to review.

    This is the primary unit of review - a single file with its content.
    Can be created from a local file, git diff, or GitHub API response.

    Attributes:
        path: File path (relative to repo root)
        content: Full file content
        language: Programming language (inferred from extension)
    """

    path: str
    content: str
    language: str = ""

    def __post_init__(self):
        """Infer language from file extension if not provided."""
        if not self.language:
            self.language = self._infer_language()

    def _infer_language(self) -> str:
        """Infer programming language from file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".sh": "bash",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".xml": "xml",
            ".html": "html",
            ".css": "css",
            ".sql": "sql",
            ".md": "markdown",
        }
        for ext, lang in ext_map.items():
            if self.path.endswith(ext):
                return lang
        return "text"

    @property
    def is_python(self) -> bool:
        """Check if file is Python."""
        return self.language == "python"

    @property
    def is_reviewable(self) -> bool:
        """Check if file should be reviewed (code files only)."""
        non_reviewable = {"json", "yaml", "xml", "html", "css", "markdown", "text"}
        return self.language not in non_reviewable

    @property
    def line_count(self) -> int:
        """Number of lines in the file."""
        return len(self.content.splitlines())

    @classmethod
    def from_diff(cls, diff: "FileDiff") -> "CodeFile":
        """Create CodeFile from a FileDiff."""
        return cls(
            path=diff.filename,
            content=diff.new_content or diff.patch,
        )


@dataclass
class FileDiff:
    """
    Changes to a file in a diff or PR.

    Represents the delta between two versions of a file.

    Attributes:
        filename: Path to the file
        status: Change type (added, removed, modified, renamed)
        additions: Number of lines added
        deletions: Number of lines deleted
        patch: The diff patch content
        new_content: Full content of the new version (if available)
        previous_filename: Original filename if renamed
    """

    filename: str
    status: str  # added, removed, modified, renamed
    additions: int = 0
    deletions: int = 0
    patch: str = ""
    new_content: Optional[str] = None
    previous_filename: Optional[str] = None

    @property
    def is_python(self) -> bool:
        """Check if file is Python."""
        return self.filename.endswith(".py")

    @property
    def total_changes(self) -> int:
        """Total lines changed."""
        return self.additions + self.deletions

    @property
    def is_added(self) -> bool:
        """Check if file was added."""
        return self.status == "added"

    @property
    def is_removed(self) -> bool:
        """Check if file was removed."""
        return self.status == "removed"

    @property
    def is_modified(self) -> bool:
        """Check if file was modified."""
        return self.status == "modified"

    @property
    def is_renamed(self) -> bool:
        """Check if file was renamed."""
        return self.status == "renamed"


@dataclass
class PullRequestInfo:
    """
    Metadata about a pull request.

    Contains information needed to review and comment on a PR.

    Attributes:
        number: PR number
        title: PR title
        body: PR description
        head_sha: SHA of the head commit
        base_sha: SHA of the base commit
        base_branch: Target branch (e.g., "main")
        head_branch: Source branch (e.g., "feature/xyz")
        author: PR author username
        files: List of changed files
    """

    number: int
    title: str
    body: str
    head_sha: str
    base_sha: str
    base_branch: str
    head_branch: str
    author: str
    files: List[FileDiff] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        """Total lines changed across all files."""
        return sum(f.total_changes for f in self.files)

    @property
    def python_files(self) -> List[FileDiff]:
        """Get only Python files."""
        return [f for f in self.files if f.is_python]

    @property
    def file_count(self) -> int:
        """Number of files changed."""
        return len(self.files)

    @property
    def is_large(self) -> bool:
        """Check if PR is considered large (>500 lines or >10 files)."""
        return self.total_changes > 500 or self.file_count > 10
