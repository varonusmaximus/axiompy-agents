"""AxiomPy Reasoning Components

This module provides AI/LLM-powered reasoning and intelligence features, including:
- Provider-agnostic AI client for multiple LLM backends
- Metadata schema for self-describing tools
- AI agents for intelligent query routing and execution
- Prompt builders for dynamic prompt construction

Features:
- Works with Ollama, OpenAI, Anthropic, and other HTTP-based AI services
- No external LLM library dependencies (uses axiompy.io.HTTPClient)
- Type-safe with full type hints
- Production-ready with comprehensive error handling

Architecture:
    reasoning/
    ├── metadata.py          # DatasetMetadata and schema definitions
    ├── metadata_helpers.py  # Utilities for working with metadata
    ├── client.py            # AIClient (provider-agnostic LLM interface)
    ├── factory.py           # AIClientFactory for creating clients
    ├── prompts.py           # DynamicPromptBuilder for prompt construction
    ├── agents/              # AI agents for reasoning
    │   └── query.py         # QueryAgent for intelligent query routing
    └── providers/           # LLM provider implementations
        ├── base.py          # ProviderConfig abstract base
        ├── ollama.py        # Ollama local LLM provider
        ├── openai.py        # OpenAI API provider
        └── anthropic.py     # Anthropic API provider
"""

from axiompy.reasoning.agents.query import QueryAgent
from axiompy.reasoning.base import BaseDatasetService
from axiompy.reasoning.client import AIClient
from axiompy.reasoning.adapter import ReasoningLLMAdapter
from axiompy.reasoning.rag_llm_adapter import ReasoningAdapter
from axiompy.reasoning.factory import ReasoningFactory
from axiompy.reasoning.settings import ReasoningSettings
from axiompy.reasoning.metadata import (
    DatasetMetadata,
    ExampleMetadata,
    ScopeMetadata,
    TableSchemaMetadata,
)
from axiompy.reasoning.types import ReasoningProvider

__all__ = [
    "BaseDatasetService",
    "AIClient",
    "ReasoningFactory",
    "ReasoningSettings",
    "ReasoningLLMAdapter",
    "ReasoningAdapter",
    "ReasoningProvider",
    "DatasetMetadata",
    "ScopeMetadata",
    "TableSchemaMetadata",
    "ExampleMetadata",
    "QueryAgent",
]
