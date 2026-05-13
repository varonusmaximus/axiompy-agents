"""LLM Provider Adapters.

Implementations of the LLMProvider port.

Available:
- ReasoningAdapter: Wraps axiompy.reasoning.AIClient for any provider

Testing:
- MockLLMProvider: In adapters/mocks.py
"""

from axiompy.agents.rag.adapters.llm.reasoning_adapter import ReasoningAdapter

__all__ = ["ReasoningAdapter"]
