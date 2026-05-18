"""Tests for ReasoningLLMAdapter."""

from unittest.mock import MagicMock

from axiompy.kernel.domain.models import Message
from axiompy.reasoning.adapter import ReasoningLLMAdapter


class TestReasoningLLMAdapter:
    def test_complete_returns_content(self) -> None:
        client = MagicMock()
        client.generate_completion.return_value = "Hello world"
        adapter = ReasoningLLMAdapter(client)
        result = adapter.complete([Message(role="user", content="Hi")], [])
        assert result.content == "Hello world"
        client.generate_completion.assert_called_once()

    def test_parse_tool_call(self) -> None:
        client = MagicMock()
        client.generate_completion.return_value = (
            'Thinking...\n<tool_call>{"id":"1","name":"search","arguments":{"q":"x"}}</tool_call>'
        )
        adapter = ReasoningLLMAdapter(client)
        result = adapter.complete([Message(role="user", content="search")], [])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"q": "x"}
