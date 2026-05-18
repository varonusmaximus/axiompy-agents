"""Bridge axiompy.reasoning AIClient to kernel LLMPort."""

from __future__ import annotations

import json
import re
from typing import List

from axiompy.kernel.domain.models import LLMCompletion, Message, ToolCall, ToolSpec
from axiompy.reasoning.client import AIClient


class ReasoningLLMAdapter:
    """Implements LLMPort using AIClient.generate_completion."""

    def __init__(self, client: AIClient) -> None:
        self._client = client

    def complete(
        self,
        messages: List[Message],
        tools: List[ToolSpec],
    ) -> LLMCompletion:
        prompt = self._format_prompt(messages, tools)
        raw = self._client.generate_completion(prompt)
        return self._parse_response(raw)

    def stream(self, messages: List[Message], tools: List[ToolSpec]):
        completion = self.complete(messages, tools)
        yield completion.content

    def _format_prompt(self, messages: List[Message], tools: List[ToolSpec]) -> str:
        lines = []
        if tools:
            lines.append("Available tools (JSON):")
            lines.append(
                json.dumps(
                    [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                        for t in tools
                    ]
                )
            )
        for msg in messages:
            lines.append(f"{msg.role}: {msg.content}")
        lines.append("assistant:")
        return "\n".join(lines)

    def _parse_response(self, raw: str) -> LLMCompletion:
        tool_calls: List[ToolCall] = []
        content = raw

        tool_match = re.search(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
            raw,
            re.DOTALL,
        )
        if tool_match:
            try:
                data = json.loads(tool_match.group(1))
                tool_calls.append(
                    ToolCall(
                        id=data.get("id", "call_1"),
                        name=data["name"],
                        arguments=data.get("arguments", {}),
                    )
                )
                content = raw[: tool_match.start()].strip()
            except (json.JSONDecodeError, KeyError):
                pass

        return LLMCompletion(content=content or raw, tool_calls=tool_calls)
