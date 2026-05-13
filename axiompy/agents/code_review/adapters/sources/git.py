"""Git Source - Read code from local git repository.

Implements CodeSource for reviewing git diffs and commits.
"""

import contextlib
import subprocess
from pathlib import Path
from typing import List, Optional

from ...domain.models import CodeFile, FileDiff, PullRequestInfo


class GitError(Exception):
    """Error running git command."""

    pass


class GitSource:
    """
    Read code from local git repository.

    Implements CodeSource port for CLI use case with git integration.

    Example:
        source = GitSource(repo_path=".")

        # Review staged changes
        diffs = source.get_diff("HEAD", "--staged")

        # Review last commit
        diffs = source.get_diff("HEAD~1", "HEAD")
    """

    def __init__(self, repo_path: str = "."):
        """
        Initialize git source.

        Args:
            repo_path: Path to git repository root
        """
        self.repo_path = Path(repo_path).resolve()
        self._verify_git_repo()

    def _verify_git_repo(self) -> None:
        """Verify the path is a git repository."""
        git_dir = self.repo_path / ".git"
        if not git_dir.exists():
            raise GitError(f"Not a git repository: {self.repo_path}")

    def get_files(self, paths: List[str]) -> List[CodeFile]:
        """
        Get files from git repository.

        Args:
            paths: List of file paths

        Returns:
            List of CodeFile objects
        """
        files = []
        for path_str in paths:
            path = self.repo_path / path_str
            if path.is_file():
                content = path.read_text(encoding="utf-8")
                files.append(CodeFile(path=path_str, content=content))
        return files

    def get_file_content(self, path: str, ref: Optional[str] = None) -> str:
        """
        Get file content, optionally at a specific ref.

        Args:
            path: Path to file
            ref: Git ref (commit, branch, tag)

        Returns:
            File content
        """
        if ref:
            result = self._run_git(["show", f"{ref}:{path}"])
            return result
        else:
            full_path = self.repo_path / path
            return full_path.read_text(encoding="utf-8")

    def get_diff(self, base: str, head: str) -> List[FileDiff]:
        """
        Get diff between two refs.

        Args:
            base: Base ref (e.g., "main", "HEAD~1")
            head: Head ref (e.g., "HEAD", "--staged")

        Returns:
            List of FileDiff objects
        """
        # Handle special case for staged changes
        if head in ("--staged", "staged"):
            return self._get_staged_diff()

        # Get list of changed files
        diff_output = self._run_git(["diff", "--name-status", base, head])

        files = []
        for line in diff_output.strip().split("\n"):
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            status_char = parts[0][0]
            filename = parts[-1]

            status_map = {
                "A": "added",
                "D": "removed",
                "M": "modified",
                "R": "renamed",
            }
            status = status_map.get(status_char, "modified")

            # Get the patch
            patch = self._run_git(["diff", base, head, "--", filename])

            # Get new content if file exists
            new_content = None
            if status != "removed":
                with contextlib.suppress(Exception):
                    new_content = self.get_file_content(filename, head)

            # Count additions/deletions
            additions = patch.count("\n+") - patch.count("\n+++")
            deletions = patch.count("\n-") - patch.count("\n---")

            files.append(
                FileDiff(
                    filename=filename,
                    status=status,
                    additions=max(0, additions),
                    deletions=max(0, deletions),
                    patch=patch,
                    new_content=new_content,
                )
            )

        return files

    def get_staged_files(self) -> List[FileDiff]:
        """Get currently staged files."""
        return self._get_staged_diff()

    def get_pull_request(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> PullRequestInfo:
        """
        Not implemented for local git.

        Use GitHubSource for PR functionality.
        """
        raise NotImplementedError("GitSource doesn't support PRs. Use GitHubSource instead.")

    def _get_staged_diff(self) -> List[FileDiff]:
        """Get diff for staged changes."""
        # Get list of staged files
        staged_output = self._run_git(["diff", "--name-status", "--cached"])

        files = []
        for line in staged_output.strip().split("\n"):
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            status_char = parts[0][0]
            filename = parts[-1]

            status_map = {
                "A": "added",
                "D": "removed",
                "M": "modified",
                "R": "renamed",
            }
            status = status_map.get(status_char, "modified")

            # Get the staged patch
            patch = self._run_git(["diff", "--cached", "--", filename])

            # Get staged content
            new_content = None
            if status != "removed":
                with contextlib.suppress(Exception):
                    new_content = self._run_git(["show", f":0:{filename}"])

            additions = patch.count("\n+") - patch.count("\n+++")
            deletions = patch.count("\n-") - patch.count("\n---")

            files.append(
                FileDiff(
                    filename=filename,
                    status=status,
                    additions=max(0, additions),
                    deletions=max(0, deletions),
                    patch=patch,
                    new_content=new_content,
                )
            )

        return files

    def _run_git(self, args: List[str]) -> str:
        """Run a git command and return output."""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise GitError(f"Git command failed: {e.stderr}")
