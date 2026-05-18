"""Ollama Provider Configuration - Local LLM inference

Configuration for Ollama, which runs LLMs locally without cloud dependencies.

Features:
- Local inference (no data sent to cloud)
- Supports multiple models (Mistral, Llama, etc.)
- Streaming responses
- No authentication required

Setup:
    1. Install Ollama from https://ollama.ai
    2. Pull a model: `ollama pull mistral`
    3. Start server: `ollama serve` (default: http://localhost:11434)
"""

from __future__ import annotations

from typing import Any

from axiompy.reasoning.providers.base import ProviderConfig


class OllamaProviderConfig(ProviderConfig):
    """Configuration for Ollama local LLM service."""

    @staticmethod
    def format_prompt(prompt_dict: dict[str, str]) -> str:
        """
        Format prompt for Ollama.

        Ollama expects a single prompt string. We combine system and user
        messages into a natural prompt format.

        Args:
            prompt_dict: Dictionary with "system" and "user" keys

        Returns:
            Combined prompt string
        """
        system = prompt_dict.get("system", "")
        user = prompt_dict.get("user", "")

        if system and user:
            return f"{system}\n\n{user}"
        elif user:
            return user
        else:
            return system

    @staticmethod
    def build_payload(
        formatted_prompt: str,
        model: str,
        endpoint: str,
        **options: Any,
    ) -> dict[str, Any]:
        """
        Build Ollama API request payload.

        Args:
            formatted_prompt: Combined prompt string from format_prompt()
            model: Model name (e.g., "mistral", "llama2")
            endpoint: Ollama server endpoint (e.g., "http://localhost:11434")
            **options: Additional options:
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Max response length (default: 500)
                - top_p: Nucleus sampling (default: 0.9)
                - top_k: Top-k sampling (default: 40)

        Returns:
            Dictionary with keys: model, prompt, temperature, top_p, top_k
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": formatted_prompt,
            "temperature": options.get("temperature", 0.7),
            "top_p": options.get("top_p", 0.9),
            "top_k": options.get("top_k", 40),
            "stream": False,  # Disable streaming for simplicity
        }

        if "max_tokens" in options:
            payload["num_predict"] = options["max_tokens"]

        return payload

    @staticmethod
    def parse_response(response_json: dict[str, Any]) -> str:
        """
        Parse Ollama response.

        Ollama returns: {"response": "generated text", "model": "...", ...}

        Args:
            response_json: Response from Ollama API

        Returns:
            Generated text

        Raises:
            KeyError: If "response" field missing
        """
        if "response" not in response_json:
            raise KeyError(f"Ollama response missing 'response' field: {response_json}")

        return response_json["response"]
