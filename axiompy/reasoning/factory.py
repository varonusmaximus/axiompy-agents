"""ReasoningFactory - Factory for creating AI client instances."""

from typing import Any, Optional

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

        match provider:
            case ReasoningProvider.OLLAMA:
                model = model or "mistral"
                endpoint = endpoint or "http://localhost:11434/api/generate"
            case ReasoningProvider.OPENAI:
                model = model or "gpt-3.5-turbo"
                endpoint = endpoint or "https://api.openai.com/v1/chat/completions"
            case ReasoningProvider.ANTHROPIC:
                model = model or "claude-2"
                endpoint = endpoint or "https://api.anthropic.com/v1/complete"
            case _:
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
    def create_mock(
        responses: Optional[list[str]] = None,
        provider: ReasoningProvider = ReasoningProvider.OLLAMA,
    ) -> AIClient:
        """
        Create a mock AIClient for testing.

        Args:
            responses: Canned completion strings (cycles on repeated calls)
            provider: Provider enum used for client metadata

        Returns:
            AIClient with mocked generate_completion
        """
        canned = list(responses or ["Mock completion."])
        client = ReasoningFactory.create(
            provider,
            settings=ReasoningSettings(model="mock-model", endpoint="http://mock"),
        )
        call_index = {"i": 0}

        def _fake_completion(*_args: Any, **_kwargs: Any) -> str:
            text = canned[call_index["i"] % len(canned)]
            call_index["i"] += 1
            return text

        client.generate_completion = _fake_completion  # type: ignore[method-assign]
        return client
