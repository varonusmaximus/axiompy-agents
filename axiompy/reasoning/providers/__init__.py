"""AI Provider Configurations - Support for multiple LLM backends

This module provides provider-specific formatting and configuration for various
AI/LLM services. Each provider handles the specific API format and requirements
of their service.

Supported Providers:
- Ollama: Local LLM inference
- OpenAI: OpenAI GPT models
- Anthropic: Anthropic Claude models

Adding New Providers:
1. Create a new module (e.g., providers/gemini.py)
2. Implement ProviderConfig abstract base
3. Add to get_provider() factory function
"""

from axiompy.reasoning.providers.base import ProviderConfig
from axiompy.reasoning.providers.factory import get_provider, list_providers

__all__ = [
    "ProviderConfig",
    "get_provider",
    "list_providers",
]
