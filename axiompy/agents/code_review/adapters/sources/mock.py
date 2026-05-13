"""Mock Source - For testing.

Implements CodeSource with configurable responses.
"""

from typing import Dict, List, Optional, Tuple

from ...domain.models import CodeFile, FileDiff, PullRequestInfo


class MockCodeSource:
    """
    Mock code source for testing.

    Allows setting up predefined files and PRs for testing.

    Example:
        source = MockCodeSource()
        source.set_file("main.py", "print('hello')")
        source.set_pr("owner", "repo", 1, mock_pr)

        files = source.get_files(["main.py"])
    """

    def __init__(self):
        """Initialize mock source."""
        self._files: Dict[str, str] = {}
        self._prs: Dict[Tuple[str, str, int], PullRequestInfo] = {}
        self._diffs: Dict[Tuple[str, str], List[FileDiff]] = {}
        self.calls: List[Tuple[str, tuple]] = []

    def set_file(self, path: str, content: str) -> "MockCodeSource":
        """Set file content."""
        self._files[path] = content
        return self

    def set_pr(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        pr: PullRequestInfo,
    ) -> "MockCodeSource":
        """Set PR data."""
        self._prs[(owner, repo, pr_number)] = pr
        return self

    def set_diff(
        self,
        base: str,
        head: str,
        diffs: List[FileDiff],
    ) -> "MockCodeSource":
        """Set diff data."""
        self._diffs[(base, head)] = diffs
        return self

    def reset(self) -> None:
        """Reset all state."""
        self._files.clear()
        self._prs.clear()
        self._diffs.clear()
        self.calls.clear()

    def get_files(self, paths: List[str]) -> List[CodeFile]:
        """Get files."""
        self.calls.append(("get_files", (paths,)))

        files = []
        for path in paths:
            if path in self._files:
                files.append(CodeFile(path=path, content=self._files[path]))
        return files

    def get_file_content(self, path: str) -> str:
        """Get file content."""
        self.calls.append(("get_file_content", (path,)))

        if path not in self._files:
            raise FileNotFoundError(f"Mock file not found: {path}")
        return self._files[path]

    def get_diff(self, base: str, head: str) -> List[FileDiff]:
        """Get diff."""
        self.calls.append(("get_diff", (base, head)))

        key = (base, head)
        if key not in self._diffs:
            return []
        return self._diffs[key]

    def get_pull_request(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> PullRequestInfo:
        """Get PR."""
        self.calls.append(("get_pull_request", (owner, repo, pr_number)))

        key = (owner, repo, pr_number)
        if key not in self._prs:
            raise ValueError(f"Mock PR not found: {owner}/{repo}#{pr_number}")
        return self._prs[key]
