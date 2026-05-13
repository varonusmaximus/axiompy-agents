"""Webhook Adapter - FastAPI webhook handler for GitHub events.

Uses axiompy.servers.ServerFactory for server creation.

Provides webhook endpoint for GitHub PR events.
"""

import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import Optional

from axiompy.loggers import LoggerFactory
from axiompy.servers import ServerFactory, ServerSettings, ServerType

from ..adapters.analyzers import AnalyzerSettings, AnalyzerType
from ..domain.service import CodeReviewService
from ..factory import (
    CodeReviewServiceFactory,
    CodeSourceSettings,
    CodeSourceType,
    PublisherSettings,
    PublisherType,
    RulesSourceSettings,
    RulesSourceType,
)

logger = LoggerFactory.create_logger(__name__)


@dataclass
class WebhookSettings:
    """Settings for webhook service."""

    host: str = "0.0.0.0"
    port: int = 8080
    github_token: str = ""
    rules_repo: str = "varonusmaximus/axiompy"
    rules_file: str = "AGENTS.md"
    ollama_host: str = "http://localhost:11434"
    ollama_model: Optional[str] = None  # Uses DEFAULT_MODEL if not set
    webhook_secret: Optional[str] = None

    def __post_init__(self):
        if not self.github_token:
            raise ValueError("GitHub token is required")

    @classmethod
    def from_env(cls) -> "WebhookSettings":
        """Create settings from environment variables."""
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN environment variable required")

        return cls(
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "8080")),
            github_token=token,
            rules_repo=os.environ.get("RULES_REPO", "varonusmaximus/axiompy"),
            rules_file=os.environ.get("RULES_FILE", "AGENTS.md"),
            ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=os.environ.get("OLLAMA_MODEL"),  # Uses DEFAULT_MODEL if not set
            webhook_secret=os.environ.get("WEBHOOK_SECRET"),
        )


class WebhookService:
    """
    Webhook service using axiompy.servers.ServerFactory.

    Example:
        service = WebhookService.create()
        service.run()
    """

    def __init__(
        self,
        server,
        settings: WebhookSettings,
        review_service: CodeReviewService,
    ):
        """
        Initialize webhook service.

        Args:
            server: Server from ServerFactory
            settings: Webhook configuration
            review_service: Domain service for reviews
        """
        self._server = server
        self._settings = settings
        self._review_service = review_service
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Register webhook routes."""

        @self._server.route("/health", methods=["GET"])
        def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "rules_repo": self._settings.rules_repo,
                "ollama_host": self._settings.ollama_host,
                "ollama_model": self._settings.ollama_model,
            }

        @self._server.route("/webhook", methods=["POST"])
        def webhook(data: dict):
            """Handle GitHub webhook events."""
            action = data.get("action")

            # Handle ping event
            if data.get("zen"):  # GitHub sends 'zen' in ping
                return {"status": "pong"}

            # Only process relevant PR actions
            if action not in ("opened", "synchronize", "reopened"):
                return {"status": "ignored", "action": action}

            # Extract PR info
            pr = data.get("pull_request", {})
            repo = data.get("repository", {})

            owner = repo.get("owner", {}).get("login")
            repo_name = repo.get("name")
            pr_number = pr.get("number")

            if not all([owner, repo_name, pr_number]):
                return {"status": "error", "message": "Missing PR info"}

            logger.info(f"Processing PR {owner}/{repo_name}#{pr_number}")

            # Run review
            try:
                result = self._review_service.review_pull_request(owner, repo_name, pr_number)

                return {
                    "status": "reviewed",
                    "pr": f"{owner}/{repo_name}#{pr_number}",
                    "score": result.score,
                    "violations": result.violation_count,
                    "approved": result.approved,
                }
            except Exception as e:
                logger.exception(f"Review failed: {e}")
                return {"status": "error", "message": str(e)}

    def _verify_signature(self, payload: bytes, signature: str) -> bool:
        """
        Verify GitHub webhook signature.

        Args:
            payload: Raw request body
            signature: X-Hub-Signature-256 header value

        Returns:
            True if signature is valid
        """
        if not self._settings.webhook_secret:
            return True  # No secret configured, skip verification

        expected = (
            "sha256="
            + hmac.new(
                self._settings.webhook_secret.encode(),
                payload,
                hashlib.sha256,
            ).hexdigest()
        )

        return hmac.compare_digest(expected, signature)

    def run(self) -> None:
        """Start the webhook server."""
        logger.info(f"Starting webhook on {self._settings.host}:{self._settings.port}")
        self._server.run(host=self._settings.host, port=self._settings.port)

    def get_app(self):
        """Get the underlying app (for uvicorn)."""
        return self._server.get_app()

    @staticmethod
    def _create_review_service(settings: WebhookSettings) -> CodeReviewService:
        """Create CodeReviewService for webhook use case using factory."""
        return CodeReviewServiceFactory.create(
            code_source_type=CodeSourceType.GITHUB,
            rules_source_type=RulesSourceType.GITHUB,
            analyzer_type=AnalyzerType.OLLAMA,
            publisher_type=PublisherType.GITHUB,
            code_source_settings=CodeSourceSettings(
                github_token=settings.github_token,
            ),
            rules_source_settings=RulesSourceSettings(
                github_token=settings.github_token,
                github_repo=settings.rules_repo,
                github_file=settings.rules_file,
            ),
            analyzer_settings=AnalyzerSettings(
                host=settings.ollama_host,
                model=settings.ollama_model,
            ),
            publisher_settings=PublisherSettings(
                github_token=settings.github_token,
            ),
        )

    @staticmethod
    def create(settings: Optional[WebhookSettings] = None) -> "WebhookService":
        """
        Create WebhookService from settings.

        Args:
            settings: Webhook settings (uses env vars if None)

        Returns:
            Configured WebhookService
        """
        if settings is None:
            settings = WebhookSettings.from_env()

        # Create server using axiompy ServerFactory
        server_settings = ServerSettings(
            host=settings.host,
            port=settings.port,
            extra_params={
                "title": "Code Review Agent",
                "description": "AI-powered code review webhook",
            },
        )
        server = ServerFactory.create(ServerType.FASTAPI, server_settings)

        # Create review service
        review_service = WebhookService._create_review_service(settings)

        return WebhookService(
            server=server,
            settings=settings,
            review_service=review_service,
        )

    @staticmethod
    def create_mock() -> "WebhookService":
        """Create mock service for testing."""
        settings = WebhookSettings(github_token="mock_token")

        server_settings = ServerSettings(host="127.0.0.1", port=8080)
        server = ServerFactory.create(ServerType.FASTAPI, server_settings)

        review_service = CodeReviewServiceFactory.create_mock()

        return WebhookService(
            server=server,
            settings=settings,
            review_service=review_service,
        )


if __name__ == "__main__":
    WebhookService.create().run()
