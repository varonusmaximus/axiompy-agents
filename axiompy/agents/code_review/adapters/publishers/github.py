"""GitHub Publisher - Post review to GitHub PR.

Implements ReviewPublisher for GitHub PR comments.
"""

import logging
from typing import List

from ...domain.results import ReviewComment, ReviewResult
from ..sources.github import GitHubSource

logger = logging.getLogger(__name__)


class GitHubPublisher:
    """
    Post review results to GitHub PR.

    Creates a PR review with inline comments for each violation.

    Example:
        publisher = GitHubPublisher(token="ghp_...")
        publisher.publish(result, {"owner": "owner", "repo": "repo", "pr_number": 123})
    """

    def __init__(self, token: str):
        """
        Initialize GitHub publisher.

        Args:
            token: GitHub personal access token
        """
        self._github = GitHubSource(token=token)

    def publish(self, result: ReviewResult, context: dict) -> None:
        """
        Post review to GitHub PR.

        Args:
            result: Review result to publish
            context: Must contain owner, repo, pr_number
        """
        owner = context.get("owner")
        repo = context.get("repo")
        pr_number = context.get("pr_number")

        if not all([owner, repo, pr_number]):
            raise ValueError("context must contain 'owner', 'repo', and 'pr_number'")

        logger.info(f"Posting review to {owner}/{repo}#{pr_number}")

        # Build review body
        body = self._build_review_body(result)

        # Determine event type
        event = result.review_event

        # Always include comments in the review body (inline comments are unreliable
        # due to GitHub's strict line position requirements for PR diffs)
        body_with_comments = self._append_comments_to_body(body, result.comments)

        self._github.create_review(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            body=body_with_comments,
            event=event,
            comments=None,
        )
        logger.info(f"Posted review: {event}, {len(result.comments)} comments in body")

    def _build_review_body(self, result: ReviewResult) -> str:
        """Build the review body markdown."""
        return result.summary

    def _append_comments_to_body(
        self,
        body: str,
        comments: List[ReviewComment],
    ) -> str:
        """Append inline comments to the review body as a fallback."""
        if not comments:
            return body

        lines = [body, "", "---", "", "### Inline Comments", ""]

        for comment in comments:
            if comment.line > 0:
                lines.append(f"**`{comment.path}:{comment.line}`**")
            else:
                lines.append(f"**`{comment.path}`**")
            lines.append("")
            lines.append(comment.body)
            lines.append("")

        return "\n".join(lines)
