"""GitHub Rules Source - Fetch rules from GitHub repository.

Implements RulesSource for fetching AGENTS.md from GitHub.
"""

import logging
from typing import Optional

from ..sources.github import GitHubNotFoundError, GitHubSource

logger = logging.getLogger(__name__)


class GitHubRulesSource:
    """
    Fetch rules from GitHub repository.

    Example:
        source = GitHubRulesSource(
            token="ghp_...",
            repo="varonusmaximus/axiompy",
        )
        content = source.get_rules()
    """

    def __init__(
        self,
        token: str,
        repo: str,
        rules_file: str = "AGENTS.md",
        overrides_file: str = ".cursorrules",
        ref: Optional[str] = None,
    ):
        """
        Initialize GitHub rules source.

        Args:
            token: GitHub personal access token
            repo: Repository in format "owner/repo"
            rules_file: Path to rules file in repo
            overrides_file: Path to overrides file in repo
            ref: Git ref (branch, tag, sha). Default: main branch
        """
        self._github = GitHubSource(token=token)

        parts = repo.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid repo format: {repo}. Expected 'owner/repo'")

        self.owner = parts[0]
        self.repo = parts[1]
        self.rules_file = rules_file
        self.overrides_file = overrides_file
        self.ref = ref

    def get_rules(self) -> str:
        """
        Fetch rules from GitHub.

        Returns:
            Rules content as string
        """
        logger.info(f"Fetching rules from {self.owner}/{self.repo}/{self.rules_file}")

        return self._github.get_file_content(
            owner=self.owner,
            repo=self.repo,
            path=self.rules_file,
            ref=self.ref,
        )

    def get_local_overrides(self) -> Optional[str]:
        """
        Fetch local overrides from repo.

        Returns:
            Overrides content or None if not found
        """
        try:
            return self._github.get_file_content(
                owner=self.owner,
                repo=self.repo,
                path=self.overrides_file,
                ref=self.ref,
            )
        except GitHubNotFoundError:
            return None
        except Exception as e:
            logger.debug(f"Could not fetch overrides: {e}")
            return None
