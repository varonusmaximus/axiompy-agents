"""Provider Factory - Get AI provider configurations by name"""

from __future__ import annotations

from typing import Type

from axiompy.reasoning.providers.anthropic import AnthropicProviderConfig
from axiompy.reasoning.providers.base import ProviderConfig
from axiompy.reasoning.providers.ollama import OllamaProviderConfig
from axiompy.reasoning.providers.openai import OpenAIProviderConfig

# Registry of available providers
_PROVIDERS: dict[str, Type[ProviderConfig]] = {
    "ollama": OllamaProviderConfig,
    "openai": OpenAIProviderConfig,
    "anthropic": AnthropicProviderConfig,
}


def get_provider(provider_name: str) -> Type[ProviderConfig]:
    """
    Get a provider configuration by name.

    Args:
        provider_name: Provider identifier ("ollama", "openai", "anthropic")

    Returns:
        ProviderConfig class for the specified provider

    Raises:
        ValueError: If provider name not recognized

    Example:
        >>> provider = get_provider("ollama")
        >>> prompt_format = provider.format_prompt({"user": "..."})
    """
    provider_name_lower = provider_name.lower()

    if provider_name_lower not in _PROVIDERS:
        available = ", ".join(_PROVIDERS.keys())
        raise ValueError(f"Unknown provider: {provider_name}. Available providers: {available}")

    return _PROVIDERS[provider_name_lower]


def list_providers() -> list[str]:
    """
    List all available provider names.

    Returns:
        Sorted list of provider identifiers

    Example:
        >>> providers = list_providers()
        >>> assert "ollama" in providers
    """
    return sorted(_PROVIDERS.keys())


def register_provider(name: str, provider_class: Type[ProviderConfig]) -> None:
    """
    Register a new provider configuration.

    Allows applications to add custom provider implementations at runtime.

    Args:
        name: Provider identifier (lowercase)
        provider_class: ProviderConfig subclass

    Raises:
        ValueError: If provider name already registered
        TypeError: If provider_class doesn't inherit from ProviderConfig

    Example:
        >>> class GeminiProviderConfig(ProviderConfig):
        ...     @staticmethod
        ...     def format_prompt(prompt_dict): ...
        ...     # ... other methods ...

        >>> register_provider("gemini", GeminiProviderConfig)
        >>> provider = get_provider("gemini")
    """
    if not issubclass(provider_class, ProviderConfig):
        raise TypeError(f"Provider class must inherit from ProviderConfig, got {provider_class}")

    name_lower = name.lower()

    if name_lower in _PROVIDERS:
        raise ValueError(f"Provider {name_lower} already registered")

    _PROVIDERS[name_lower] = provider_class
