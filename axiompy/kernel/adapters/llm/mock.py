"""Mock LLM for testing."""

from __future__ import annotations

from typing import List

from axiompy.kernel.domain.models import LLMCompletion, Message, ToolCall, ToolSpec


class MockLLMPort:
    """Returns scripted completions."""

    def __init__(
        self,
        responses: List[str] | None = None,
        tool_calls: List[List[ToolCall]] | None = None,
    ) -> None:
        self._responses = list(responses or ["Done."])
        self._tool_calls = list(tool_calls or [])
        self._index = 0

    def complete(
        self,
        messages: List[Message],
        tools: List[ToolSpec],
    ) -> LLMCompletion:
        idx = min(self._index, len(self._responses) - 1)
        content = self._responses[idx]
        calls: List[ToolCall] = []
        if self._index < len(self._tool_calls):
            calls = list(self._tool_calls[self._index])
        self._index += 1
        return LLMCompletion(content=content, tool_calls=calls)

    def stream(self, messages: List[Message], tools: List[ToolSpec]):
        completion = self.complete(messages, tools)
        yield completion.content
