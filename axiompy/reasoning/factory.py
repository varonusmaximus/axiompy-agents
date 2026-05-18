"""ReasoningFactory - Factory for creating AI client instances."""

from typing import Optional

from axiompy.io.http import HTTPClient
from axiompy.reasoning.client import AIClient
from axiompy.reasoning.settings import ReasoningSettings
from axiompy.reasoning.types import ReasoningProvider


class ReasoningFactory:
    """Factory for creating AI client instances."""

    @staticmethod
    def create(
        provider: ReasoningProvider,
        settings: Optional[ReasoningSettings] = None,
        http_client: Optional[HTTPClient] = None,
    ) -> AIClient:
        """
        Create an AIClient for the specified provider.

        Args:
            provider: ReasoningProvider enum
            settings: Optional explicit configuration
            http_client: Custom HTTPClient (optional)

        Returns:
            Configured AIClient instance
        """
        cfg = settings or ReasoningSettings()
        model = cfg.model
        endpoint = cfg.endpoint
        api_key = cfg.api_key

        if provider == ReasoningProvider.OLLAMA:
            model = model or "mistral"
            endpoint = endpoint or "http://localhost:11434/api/generate"
        elif provider == ReasoningProvider.OPENAI:
            model = model or "gpt-3.5-turbo"
            endpoint = endpoint or "https://api.openai.com/v1/chat/completions"
        elif provider == ReasoningProvider.ANTHROPIC:
            model = model or "claude-2"
            endpoint = endpoint or "https://api.anthropic.com/v1/complete"
        else:
            raise ValueError(f"Unknown provider: {provider}")

        if not model:
            raise ValueError(f"Model must be specified for provider {provider}")
        if not endpoint:
            raise ValueError(f"Endpoint must be specified for provider {provider}")

        return AIClient(
            provider=provider.value,
            model=model,
            endpoint=endpoint,
            api_key=api_key,
            http_client=http_client,
            cache_size=cfg.cache_size,
        )

    @staticmethod
    def create_ollama(
        model: str = "mistral",
        endpoint: str = "http://localhost:11434/api/generate",
        http_client: Optional[HTTPClient] = None,
        settings: Optional[ReasoningSettings] = None,
    ) -> AIClient:
        cfg = settings or ReasoningSettings(model=model, endpoint=endpoint)
        return ReasoningFactory.create(
            ReasoningProvider.OLLAMA,
            settings=cfg,
            http_client=http_client,
        )

    @staticmethod
    def create_openai(
        api_key: str,
        model: str = "gpt-3.5-turbo",
        endpoint: str = "https://api.openai.com/v1/chat/completions",
        http_client: Optional[HTTPClient] = None,
        settings: Optional[ReasoningSettings] = None,
    ) -> AIClient:
        cfg = settings or ReasoningSettings(model=model, endpoint=endpoint, api_key=api_key)
        return ReasoningFactory.create(
            ReasoningProvider.OPENAI,
            settings=cfg,
            http_client=http_client,
        )

    @staticmethod
    def create_anthropic(
        api_key: str,
        model: str = "claude-2",
        endpoint: str = "https://api.anthropic.com/v1/complete",
        http_client: Optional[HTTPClient] = None,
        settings: Optional[ReasoningSettings] = None,
    ) -> AIClient:
        cfg = settings or ReasoningSettings(model=model, endpoint=endpoint, api_key=api_key)
        return ReasoningFactory.create(
            ReasoningProvider.ANTHROPIC,
            settings=cfg,
            http_client=http_client,
        )
