"""Anthropic Provider Configuration - Anthropic Claude models

Configuration for Anthropic's Claude API, supporting Claude 2, Claude 3, etc.

Features:
- High-quality language models with strong reasoning
- Prompt caching for cost optimization
- Multiple model variants (Claude 1, 2, 3, etc.)
- Constitutional AI for safety

Setup:
    1. Get API key from https://console.anthropic.com
    2. Set environment: export ANTHROPIC_API_KEY="sk-ant-..."
    3. See axiompy.secrets for secure key management
"""

from __future__ import annotations

from typing import Any

from axiompy.reasoning.providers.base import ProviderConfig


class AnthropicProviderConfig(ProviderConfig):
    """Configuration for Anthropic Claude API service."""

    @staticmethod
    def format_prompt(prompt_dict: dict[str, str]) -> str:
        """
        Format prompt for Anthropic's Completions API.

        Anthropic uses a combined prompt string with special markers.

        Args:
            prompt_dict: Dictionary with "system" and "user" keys

        Returns:
            Combined prompt string with Anthropic markers
        """
        system = prompt_dict.get("system", "")
        user = prompt_dict.get("user", "")

        # Build prompt using Anthropic's format
        prompt_parts = []

        if system:
            prompt_parts.append(system)

        if user:
            prompt_parts.append(f"\n\nHuman: {user}\n\nAssistant:")

        return "".join(prompt_parts)

    @staticmethod
    def build_payload(
        formatted_prompt: str,
        model: str,
        endpoint: str,
        **options: Any,
    ) -> dict[str, Any]:
        """
        Build Anthropic API request payload.

        Args:
            formatted_prompt: Combined prompt string from format_prompt()
            model: Model name (e.g., "claude-2", "claude-3-opus")
            endpoint: API endpoint (usually "https://api.anthropic.com/v1/complete")
            **options: Additional options:
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Max response length (default: 500)
                - top_p: Nucleus sampling (default: 1.0)
                - top_k: Top-k sampling (default: 0, disabled)

        Returns:
            Dictionary with Anthropic API request format
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": formatted_prompt,
            "max_tokens_to_sample": options.get("max_tokens", 500),
            "temperature": options.get("temperature", 0.7),
            "top_p": options.get("top_p", 1.0),
            "top_k": options.get("top_k", 0),
        }

        return payload

    @staticmethod
    def parse_response(response_json: dict[str, Any]) -> str:
        """
        Parse Anthropic Completions API response.

        Anthropic returns: {"completion": "generated text", "stop_reason": "..."}

        Args:
            response_json: Response from Anthropic API

        Returns:
            Generated text

        Raises:
            KeyError: If "completion" field missing
        """
        if "completion" not in response_json:
            raise KeyError(f"Anthropic response missing 'completion': {response_json}")

        return response_json["completion"]
