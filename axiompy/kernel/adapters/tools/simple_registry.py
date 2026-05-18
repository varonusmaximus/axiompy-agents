"""In-memory tool registry."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from axiompy.kernel.domain.errors import KernelToolError
from axiompy.kernel.domain.models import ToolCall, ToolSpec
from axiompy.kernel.domain.ports import ToolRegistry


class SimpleToolRegistry(ToolRegistry):
    """Registry mapping tool names to callables."""

    def __init__(self) -> None:
        self._tools: Dict[str, tuple[ToolSpec, Callable[..., Any]]] = {}

    def register(
        self,
        spec: ToolSpec,
        handler: Callable[..., Any],
    ) -> "SimpleToolRegistry":
        self._tools[spec.name] = (spec, handler)
        return self

    def list_tools(self) -> List[ToolSpec]:
        return [spec for spec, _ in self._tools.values()]

    def invoke(self, tool_call: ToolCall) -> str:
        if tool_call.name not in self._tools:
            raise KernelToolError(f"Unknown tool: {tool_call.name}")
        _, handler = self._tools[tool_call.name]
        try:
            result = handler(**tool_call.arguments)
        except Exception as exc:
            raise KernelToolError(f"Tool {tool_call.name} failed: {exc}") from exc
        if isinstance(result, str):
            return result
        return json.dumps(result)
