"""Base Provider Configuration - Abstract interface for AI service providers

This module defines the ProviderConfig abstract base class that all
provider implementations must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Union


class ProviderConfig(ABC):
    """
    Abstract base class for AI provider configurations.

    Providers handle the specific formatting and API requirements of
    different AI services. This enables the same application code to work
    with different providers (Ollama, OpenAI, Anthropic, etc.) by only
    changing the provider configuration.

    Architecture:
    - Domain prompts (WHAT to say) come from the application
    - Provider formatting (HOW to say it) happens here
    - This separation enables prompt reuse across providers

    Example:
        # Same prompt content for all providers
        prompt = {"system": "You are a helpful SQL expert", "user": "..."}

        # Different formatting per provider
        ollama_format = OllamaProviderConfig.format_prompt(prompt)
        openai_format = OpenAIProviderConfig.format_prompt(prompt)
    """

    @staticmethod
    @abstractmethod
    def format_prompt(prompt_dict: dict[str, str]) -> Union[str, list[dict], dict]:
        """
        Format a domain prompt for this provider's API.

        Converts application prompt dict into provider-specific format.

        Args:
            prompt_dict: Dictionary with prompt components
                - "system": System prompt (role/behavior)
                - "user": User message
                - Other keys: provider-specific (e.g., "examples")

        Returns:
            Provider-specific prompt format:
            - str: Plain text prompt (e.g., Ollama)
            - list[dict]: Message array (e.g., OpenAI)
            - dict: Formatted dict with provider fields

        Raises:
            ValueError: If required prompt components missing

        Example:
            prompt = {
                "system": "You are a SQL expert",
                "user": "How many users in database?"
            }
            # Ollama: str with combined messages
            # OpenAI: [{"role": "system", ...}, {"role": "user", ...}]
        """
        ...

    @staticmethod
    @abstractmethod
    def build_payload(
        formatted_prompt: Union[str, list[dict], dict],
        model: str,
        endpoint: str,
        **options: Any,
    ) -> dict[str, Any]:
        """
        Build provider-specific API request payload.

        Constructs the complete request body for the provider's API,
        including model, prompt, and other parameters.

        Args:
            formatted_prompt: Output from format_prompt()
            model: Model name/ID for the provider
            endpoint: API endpoint URL
            **options: Additional parameters:
                - temperature: Sampling temperature (0.0-1.0)
                - max_tokens: Max response length
                - top_p: Nucleus sampling parameter
                - Other provider-specific options

        Returns:
            Dictionary ready to send as HTTP request body

        Example:
            payload = OllamaProviderConfig.build_payload(
                formatted_prompt="Use SQL to answer...",
                model="mistral",
                endpoint="http://localhost:11434",
                temperature=0.7,
                max_tokens=500
            )
            # Returns: {"model": "mistral", "prompt": "...", "temperature": 0.7, ...}
        """
        ...

    @staticmethod
    @abstractmethod
    def parse_response(response_json: dict[str, Any]) -> str:
        """
        Parse provider response and extract generated text.

        Different providers return responses in different formats.
        This extracts the generated text from the provider's response.

        Args:
            response_json: Parsed JSON response from provider's API

        Returns:
            Generated text string

        Raises:
            KeyError: If response format unexpected
            ValueError: If response contains error

        Example:
            # Ollama response: {"response": "SELECT..."}
            text = OllamaProviderConfig.parse_response({"response": "SELECT..."})

            # OpenAI response: {"choices": [{"message": {"content": "SELECT..."}}]}
            text = OpenAIProviderConfig.parse_response({"choices": [...]})
        """
        ...
