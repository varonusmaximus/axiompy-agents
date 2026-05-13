"""GitHub Source - Fetch code from GitHub API.

Implements CodeSource for reviewing GitHub PRs.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from ...domain.models import CodeFile, FileDiff, PullRequestInfo

logger = logging.getLogger(__name__)


class GitHubError(Exception):
    """Base exception for GitHub errors."""

    pass


class GitHubAuthError(GitHubError):
    """Authentication failed."""

    pass


class GitHubNotFoundError(GitHubError):
    """Resource not found."""

    pass


@dataclass
class GitHubSourceSettings:
    """Settings for GitHub source."""

    token: str
    base_url: str = "https://api.github.com"
    timeout: int = 30

    def __post_init__(self):
        if not self.token:
            raise ValueError("GitHub token is required")


class GitHubSource:
    """
    Fetch code from GitHub API.

    Implements CodeSource port for webhook/action use case.

    Example:
        source = GitHubSource(token="ghp_...")
        pr = source.get_pull_request("owner", "repo", 123)
        files = [CodeFile.from_diff(f) for f in pr.files]
    """

    def __init__(
        self,
        token: str,
        base_url: str = "https://api.github.com",
        timeout: int = 30,
    ):
        """
        Initialize GitHub source.

        Args:
            token: GitHub personal access token
            base_url: GitHub API base URL
            timeout: Request timeout in seconds
        """
        self._settings = GitHubSourceSettings(
            token=token,
            base_url=base_url.rstrip("/"),
            timeout=timeout,
        )
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def get_files(self, paths: List[str]) -> List[CodeFile]:
        """
        Get files from repository.

        Note: This requires owner/repo context which isn't available here.
        Use get_pull_request() for PR files instead.
        """
        raise NotImplementedError(
            "GitHubSource.get_files() requires repo context. Use get_pull_request() instead."
        )

    def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: Optional[str] = None,
    ) -> str:
        """
        Get content of a file from repository.

        Args:
            owner: Repository owner
            repo: Repository name
            path: Path to file in repo
            ref: Git ref (branch, tag, sha)

        Returns:
            File content as string
        """
        import requests

        url = f"{self._settings.base_url}/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": ref} if ref else {}

        response = requests.get(
            url,
            headers={**self._headers, "Accept": "application/vnd.github.v3.raw"},
            params=params,
            timeout=self._settings.timeout,
        )

        if response.status_code == 401:
            raise GitHubAuthError("Invalid GitHub token")
        if response.status_code == 404:
            raise GitHubNotFoundError(f"File not found: {path}")

        response.raise_for_status()
        return response.text

    def get_diff(self, base: str, head: str) -> List[FileDiff]:
        """
        Not directly supported by GitHubSource.

        Use get_pull_request() for PR diffs.
        """
        raise NotImplementedError(
            "GitHubSource.get_diff() requires repo context. Use get_pull_request() instead."
        )

    def get_pull_request(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        include_files: bool = True,
    ) -> PullRequestInfo:
        """
        Get pull request information with files.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: PR number
            include_files: Whether to fetch file list

        Returns:
            PullRequestInfo with files
        """
        import requests

        # Get PR info
        url = f"{self._settings.base_url}/repos/{owner}/{repo}/pulls/{pr_number}"
        response = requests.get(
            url,
            headers=self._headers,
            timeout=self._settings.timeout,
        )

        if response.status_code == 401:
            raise GitHubAuthError("Invalid GitHub token")
        if response.status_code == 404:
            raise GitHubNotFoundError(f"PR #{pr_number} not found")

        response.raise_for_status()
        pr_data = response.json()

        # Get files
        files: List[FileDiff] = []
        if include_files:
            files = self._get_pr_files(owner, repo, pr_number)

        return PullRequestInfo(
            number=pr_data["number"],
            title=pr_data["title"],
            body=pr_data.get("body") or "",
            head_sha=pr_data["head"]["sha"],
            base_sha=pr_data["base"]["sha"],
            base_branch=pr_data["base"]["ref"],
            head_branch=pr_data["head"]["ref"],
            author=pr_data["user"]["login"],
            files=files,
        )

    def _get_pr_files(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> List[FileDiff]:
        """Get files changed in a PR."""
        import requests

        files = []
        page = 1

        while True:
            url = f"{self._settings.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/files"
            response = requests.get(
                url,
                headers=self._headers,
                params={"per_page": 100, "page": page},
                timeout=self._settings.timeout,
            )

            response.raise_for_status()
            page_files = response.json()

            if not page_files:
                break

            for f in page_files:
                files.append(
                    FileDiff(
                        filename=f["filename"],
                        status=f["status"],
                        additions=f.get("additions", 0),
                        deletions=f.get("deletions", 0),
                        patch=f.get("patch", ""),
                        previous_filename=f.get("previous_filename"),
                    )
                )

            page += 1
            if len(page_files) < 100:
                break

        return files

    def create_review(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str,
        event: str = "COMMENT",
        comments: Optional[List[dict]] = None,
    ) -> dict:
        """
        Create a review on a PR.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: PR number
            body: Review body
            event: Review event (APPROVE, REQUEST_CHANGES, COMMENT)
            comments: List of inline comments

        Returns:
            Review response data
        """
        import requests

        url = f"{self._settings.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/reviews"

        data = {
            "body": body,
            "event": event,
        }

        if comments:
            data["comments"] = comments

        response = requests.post(
            url,
            headers=self._headers,
            json=data,
            timeout=self._settings.timeout,
        )

        if not response.ok:
            # Log the actual error from GitHub for debugging
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"GitHub API error {response.status_code}: {response.text}")
            logger.error(f"Request data: event={event}, body_length={len(body)}")

        response.raise_for_status()
        return response.json()
