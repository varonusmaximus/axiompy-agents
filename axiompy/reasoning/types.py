"""Type definitions for reasoning module."""

from enum import Enum


class ReasoningProvider(str, Enum):
    """Supported AI/LLM providers for reasoning."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
