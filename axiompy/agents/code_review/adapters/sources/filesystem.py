"""FileSystem Source - Read code from local filesystem.

Implements CodeSource for reviewing local files.
"""

import os
from pathlib import Path
from typing import List

from ...domain.models import CodeFile, FileDiff, PullRequestInfo


class FileSystemSource:
    """
    Read code files from local filesystem.

    Implements CodeSource port for CLI use case.

    Example:
        source = FileSystemSource(root="./src")
        files = source.get_files(["main.py", "utils/"])
    """

    def __init__(self, root: str = "."):
        """
        Initialize filesystem source.

        Args:
            root: Root directory for relative paths
        """
        self.root = Path(root).resolve()

    def get_files(self, paths: List[str]) -> List[CodeFile]:
        """
        Get code files from paths.

        Handles both files and directories (recursive).

        Args:
            paths: List of file or directory paths

        Returns:
            List of CodeFile objects
        """
        files = []

        for path_str in paths:
            path = self._resolve_path(path_str)

            if path.is_file():
                file = self._read_file(path)
                if file:
                    files.append(file)
            elif path.is_dir():
                for file_path in self._walk_directory(path):
                    file = self._read_file(file_path)
                    if file:
                        files.append(file)

        return files

    def get_file_content(self, path: str) -> str:
        """
        Get content of a single file.

        Args:
            path: Path to file

        Returns:
            File content

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        full_path = self._resolve_path(path)
        return full_path.read_text(encoding="utf-8")

    def get_diff(self, base: str, head: str) -> List[FileDiff]:
        """
        Not implemented for filesystem source.

        Use GitSource for diff functionality.
        """
        raise NotImplementedError("FileSystemSource doesn't support diffs. Use GitSource instead.")

    def get_pull_request(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> PullRequestInfo:
        """
        Not implemented for filesystem source.

        Use GitHubSource for PR functionality.
        """
        raise NotImplementedError("FileSystemSource doesn't support PRs. Use GitHubSource instead.")

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve path relative to root."""
        path = Path(path_str)
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def _read_file(self, path: Path) -> CodeFile | None:
        """Read a single file into CodeFile."""
        if not self._should_read(path):
            return None

        try:
            content = path.read_text(encoding="utf-8")
            # Try relative path, fall back to absolute for external files
            try:
                rel_path = str(path.relative_to(self.root))
            except ValueError:
                rel_path = str(path)
            return CodeFile(path=rel_path, content=content)
        except (UnicodeDecodeError, PermissionError):
            return None

    def _walk_directory(self, directory: Path) -> List[Path]:
        """Walk directory recursively, respecting ignores."""
        files = []

        for item in directory.rglob("*"):
            if item.is_file() and self._should_read(item):
                files.append(item)

        return files

    def _should_read(self, path: Path) -> bool:
        """Check if file should be read."""
        # Skip hidden files/directories
        if any(part.startswith(".") for part in path.parts):
            return False

        # Skip common non-code directories
        skip_dirs = {
            "node_modules",
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "env",
            "dist",
            "build",
            ".tox",
            ".pytest_cache",
        }
        if any(part in skip_dirs for part in path.parts):
            return False

        # Skip non-code files
        code_extensions = {
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".cs",
            ".cpp",
            ".c",
            ".h",
            ".swift",
            ".kt",
            ".scala",
        }
        return path.suffix in code_extensions
