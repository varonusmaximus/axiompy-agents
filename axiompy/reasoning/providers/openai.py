"""OpenAI Provider Configuration - OpenAI GPT models

Configuration for OpenAI's API, supporting GPT-3.5, GPT-4, and other models.

Features:
- State-of-the-art language models
- Chat completion API
- Multiple model variants (GPT-3.5, GPT-4, etc.)
- Function calling support

Setup:
    1. Get API key from https://platform.openai.com
    2. Set environment: export OPENAI_API_KEY="sk-..."
    3. See axiompy.secrets for secure key management
"""

from __future__ import annotations

from typing import Any, Union

from axiompy.reasoning.providers.base import ProviderConfig


class OpenAIProviderConfig(ProviderConfig):
    """Configuration for OpenAI API service."""

    @staticmethod
    def format_prompt(prompt_dict: dict[str, str]) -> list[dict[str, str]]:
        """
        Format prompt for OpenAI Chat Completion API.

        OpenAI uses a messages array with roles (system, user, assistant).

        Args:
            prompt_dict: Dictionary with "system" and "user" keys

        Returns:
            List of message dicts with "role" and "content" keys
        """
        messages: list[dict[str, str]] = []

        if "system" in prompt_dict:
            messages.append({"role": "system", "content": prompt_dict["system"]})

        if "user" in prompt_dict:
            messages.append({"role": "user", "content": prompt_dict["user"]})

        return messages

    @staticmethod
    def build_payload(
        formatted_prompt: list[dict[str, str]],
        model: str,
        endpoint: str,
        **options: Any,
    ) -> dict[str, Any]:
        """
        Build OpenAI Chat Completion API request payload.

        Args:
            formatted_prompt: Message list from format_prompt()
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            endpoint: API endpoint (usually "https://api.openai.com/v1/chat/completions")
            **options: Additional options:
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Max response length (default: 500)
                - top_p: Nucleus sampling (default: 1.0)

        Returns:
            Dictionary with OpenAI API request format
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": formatted_prompt,
            "temperature": options.get("temperature", 0.7),
            "max_tokens": options.get("max_tokens", 500),
            "top_p": options.get("top_p", 1.0),
        }

        return payload

    @staticmethod
    def parse_response(response_json: dict[str, Any]) -> str:
        """
        Parse OpenAI Chat Completion response.

        OpenAI returns: {"choices": [{"message": {"content": "..."}}], ...}

        Args:
            response_json: Response from OpenAI API

        Returns:
            Generated text from the first choice

        Raises:
            KeyError: If response format unexpected
            IndexError: If no choices in response
        """
        if "choices" not in response_json or not response_json["choices"]:
            raise KeyError(f"OpenAI response missing 'choices': {response_json}")

        first_choice = response_json["choices"][0]

        if "message" not in first_choice or "content" not in first_choice["message"]:
            raise KeyError(f"OpenAI choice missing message.content: {first_choice}")

        return first_choice["message"]["content"]
