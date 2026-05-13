"""ReasoningFactory - Factory for creating AI client instances

Provides factory methods for creating pre-configured AIClient instances
for supported providers using enum-based type safety.
"""

from typing import Optional

from axiompy.io.http import HTTPClient
from axiompy.reasoning.client import AIClient
from axiompy.reasoning.types import ReasoningProvider


class ReasoningFactory:
    """
    Factory for creating AI client instances.

    Uses enum-based provider selection for type safety and consistency
    with other AxiomPy factory patterns (DatabaseFactory, ObjectStorageFactory).
    """

    @staticmethod
    def create(
        provider: ReasoningProvider,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        http_client: Optional[HTTPClient] = None,
        **kwargs,
    ) -> AIClient:
        """
        Create an AIClient for the specified provider.

        Args:
            provider: ReasoningProvider enum (OLLAMA, OPENAI, ANTHROPIC)
            model: Model name (optional, uses provider default if not specified)
            endpoint: API endpoint (optional, uses provider default if not specified)
            api_key: API key for authentication (optional)
            http_client: Custom HTTPClient (optional)
            **kwargs: Additional arguments passed to AIClient

        Returns:
            Configured AIClient instance

        Raises:
            ValueError: If provider or model is invalid

        Example:
            from axiompy.reasoning import ReasoningFactory, ReasoningProvider

            # Create Ollama client
            client = ReasoningFactory.create(ReasoningProvider.OLLAMA)

            # Create OpenAI client with custom model
            client = ReasoningFactory.create(
                ReasoningProvider.OPENAI,
                model="gpt-4",
                api_key="sk-..."
            )
        """
        provider_lower = provider.value

        # Apply provider defaults
        if provider == ReasoningProvider.OLLAMA:
            if model is None:
                model = "mistral"
            if endpoint is None:
                endpoint = "http://localhost:11434/api/generate"

        elif provider == ReasoningProvider.OPENAI:
            if model is None:
                model = "gpt-3.5-turbo"
            if endpoint is None:
                endpoint = "https://api.openai.com/v1/chat/completions"

        elif provider == ReasoningProvider.ANTHROPIC:
            if model is None:
                model = "claude-2"
            if endpoint is None:
                endpoint = "https://api.anthropic.com/v1/complete"

        else:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Supported providers: {', '.join([p.value for p in ReasoningProvider])}"
            )

        if not model:
            raise ValueError(f"Model must be specified for provider {provider}")
        if not endpoint:
            raise ValueError(f"Endpoint must be specified for provider {provider}")

        return AIClient(
            provider=provider_lower,
            model=model,
            endpoint=endpoint,
            api_key=api_key,
            http_client=http_client,
            **kwargs,
        )

    @staticmethod
    def create_ollama(
        model: str = "mistral",
        endpoint: str = "http://localhost:11434/api/generate",
        http_client: Optional[HTTPClient] = None,
        **kwargs,
    ) -> AIClient:
        """
        Create an Ollama AI client (convenience method).

        Args:
            model: Model name (default: "mistral")
            endpoint: Ollama server endpoint (default: local)
            http_client: Custom HTTPClient (optional)
            **kwargs: Additional arguments

        Returns:
            Configured Ollama AIClient

        Example:
            >>> from axiompy.reasoning import ReasoningFactory
            >>> client = ReasoningFactory.create_ollama(model="llama2")
            >>> response = client.generate_completion("Hello")
        """
        return ReasoningFactory.create(
            ReasoningProvider.OLLAMA,
            model=model,
            endpoint=endpoint,
            http_client=http_client,
            **kwargs,
        )

    @staticmethod
    def create_openai(
        api_key: str,
        model: str = "gpt-3.5-turbo",
        endpoint: str = "https://api.openai.com/v1/chat/completions",
        http_client: Optional[HTTPClient] = None,
        **kwargs,
    ) -> AIClient:
        """
        Create an OpenAI AI client (convenience method).

        Args:
            api_key: OpenAI API key (required)
            model: Model name (default: "gpt-3.5-turbo")
            endpoint: OpenAI API endpoint (default: official)
            http_client: Custom HTTPClient (optional)
            **kwargs: Additional arguments

        Returns:
            Configured OpenAI AIClient

        Example:
            >>> from axiompy.reasoning import ReasoningFactory
            >>> client = ReasoningFactory.create_openai(
            ...     api_key="sk-...",
            ...     model="gpt-4"
            ... )
            >>> response = client.generate_completion("Hello")
        """
        return ReasoningFactory.create(
            ReasoningProvider.OPENAI,
            model=model,
            endpoint=endpoint,
            api_key=api_key,
            http_client=http_client,
            **kwargs,
        )

    @staticmethod
    def create_anthropic(
        api_key: str,
        model: str = "claude-2",
        endpoint: str = "https://api.anthropic.com/v1/complete",
        http_client: Optional[HTTPClient] = None,
        **kwargs,
    ) -> AIClient:
        """
        Create an Anthropic AI client (convenience method).

        Args:
            api_key: Anthropic API key (required)
            model: Model name (default: "claude-2")
            endpoint: Anthropic API endpoint (default: official)
            http_client: Custom HTTPClient (optional)
            **kwargs: Additional arguments

        Returns:
            Configured Anthropic AIClient

        Example:
            >>> from axiompy.reasoning import ReasoningFactory
            >>> client = ReasoningFactory.create_anthropic(
            ...     api_key="sk-ant-...",
            ...     model="claude-3-opus"
            ... )
            >>> response = client.generate_completion("Hello")
        """
        return ReasoningFactory.create(
            ReasoningProvider.ANTHROPIC,
            model=model,
            endpoint=endpoint,
            api_key=api_key,
            http_client=http_client,
            **kwargs,
        )
